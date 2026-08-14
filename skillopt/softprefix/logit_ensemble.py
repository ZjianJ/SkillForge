"""Autoregressive decoding for soft-prefix learners combined in logit space."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from skillopt.softprefix.prcb_v6 import combine_boosted_logits


class LogitBoostedPrefixGenerator:
    """Greedy decoder with one independent KV cache per soft prefix.

    For a base prefix P0 and learners Ps, the next-token logits are

        F = z(P0) + sum_s alpha_s * C(z(Ps) - z(P0)).

    All streams consume the same generated history.  The backbone parameters
    are shared; only KV caches and the short prefix embeddings are duplicated.
    """

    def __init__(
        self,
        prefix_model: Any,
        *,
        base_prefix: Any,
        learner_prefixes: list[Any],
        alphas: list[float],
        response_cache_path: str | Path | None = None,
    ) -> None:
        if len(learner_prefixes) != len(alphas):
            raise ValueError("learner_prefixes and alphas must have equal length")
        self.prefix_model = prefix_model
        self.torch = prefix_model.torch
        self.base_prefix = base_prefix.detach().to("cpu").clone()
        self.learner_prefixes = [value.detach().to("cpu").clone() for value in learner_prefixes]
        self.alphas = [float(value) for value in alphas]
        self.response_cache_path = Path(response_cache_path) if response_cache_path else None
        self._response_cache: dict[str, str] = {}
        if self.response_cache_path and self.response_cache_path.exists():
            with self.response_cache_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self._response_cache[str(row["prompt_sha256"])] = str(row["response"])

    def _install(self, value: Any) -> None:
        with self.torch.no_grad():
            self.prefix_model.prefix_embeddings.copy_(
                value.to(
                    device=self.prefix_model.device,
                    dtype=self.prefix_model.prefix_embeddings.dtype,
                )
            )

    def _initial_stream(self, input_ids: Any, attention_mask: Any, prefix: Any, insert_idx: int | None):
        self._install(prefix)
        insert_tensor = None
        if insert_idx is not None:
            insert_tensor = self.torch.tensor([insert_idx], device=self.prefix_model.device)
        embeds, full_attention, _ = self.prefix_model._with_prefix(
            input_ids,
            attention_mask,
            prefix_insert_idx=insert_tensor,
        )
        output = self.prefix_model.model(
            inputs_embeds=embeds,
            attention_mask=full_attention,
            use_cache=True,
            output_router_logits=False,
            logits_to_keep=1,
            return_dict=True,
        )
        return output.logits[:, -1, :].float(), output.past_key_values, full_attention

    def _advance_stream(self, token: Any, attention_mask: Any, cache: Any):
        output = self.prefix_model.model(
            input_ids=token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            output_router_logits=False,
            logits_to_keep=1,
            return_dict=True,
        )
        return output.logits[:, -1, :].float(), output.past_key_values

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        prefix_insert_idx: int | None = None,
    ) -> str:
        if temperature != 0:
            raise ValueError("LogitBoostedPrefixGenerator currently supports greedy decoding only")
        encoded = self.prefix_model.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.prefix_model.device)
        attention = encoded["attention_mask"].to(self.prefix_model.device)
        prefixes = [self.base_prefix, *self.learner_prefixes]
        with self.torch.inference_mode():
            streams = [
                self._initial_stream(input_ids, attention, prefix, prefix_insert_idx)
                for prefix in prefixes
            ]
            logits = [stream[0] for stream in streams]
            caches = [stream[1] for stream in streams]
            stream_attention = [stream[2] for stream in streams]
            generated: list[int] = []
            eos = self.prefix_model.tokenizer.eos_token_id
            for step in range(int(max_new_tokens)):
                combined = combine_boosted_logits(
                    self.torch,
                    logits[0],
                    logits[1:],
                    self.alphas,
                )
                token_id = int(combined.argmax(dim=-1).item())
                generated.append(token_id)
                if eos is not None and token_id == int(eos):
                    break
                if step + 1 >= int(max_new_tokens):
                    break
                token = self.torch.tensor(
                    [[token_id]], dtype=self.torch.long, device=self.prefix_model.device
                )
                next_logits = []
                for index in range(len(prefixes)):
                    stream_attention[index] = self.torch.cat(
                        [
                            stream_attention[index],
                            self.torch.ones(
                                (1, 1),
                                dtype=stream_attention[index].dtype,
                                device=stream_attention[index].device,
                            ),
                        ],
                        dim=-1,
                    )
                    value, caches[index] = self._advance_stream(
                        token,
                        stream_attention[index],
                        caches[index],
                    )
                    next_logits.append(value)
                logits = next_logits
        del streams, logits, caches, stream_attention
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return self.prefix_model.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        prefix_insert_indices: list[int | None] | None = None,
    ) -> list[str]:
        indices = prefix_insert_indices or [None] * len(prompts)
        if len(indices) != len(prompts):
            raise ValueError("prefix_insert_indices must match prompts")
        responses = []
        progress = tqdm(
            zip(prompts, indices, strict=True),
            total=len(prompts),
            desc="V6 Ensemble Generate",
            unit="ex",
        )
        for prompt, insert_idx in progress:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if digest in self._response_cache:
                responses.append(self._response_cache[digest])
                continue
            response = self.generate_from_prompt(
                prompt,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prefix_insert_idx=insert_idx,
            )
            responses.append(response)
            self._response_cache[digest] = response
            if self.response_cache_path:
                self.response_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.response_cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"prompt_sha256": digest, "response": response},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
        return responses
