"""Frozen causal LM wrapper with trainable soft-prefix embeddings."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from skillopt.softprefix.data import (
    apply_docvqa_image_budget,
    qwen_image_patch_size,
    resolve_docvqa_image_token_budget,
)


def _import_torch_and_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Soft-prefix training requires the optional dependencies "
            "`torch` and `transformers`. Install them with `pip install -e '.[softprefix]'`."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _import_torch_and_vlm_transformers():
    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as AutoVLM
            except ImportError:
                from transformers import AutoModelForCausalLM as AutoVLM
    except ImportError as exc:
        raise ImportError(
            "DocVQA soft-prefix training requires the optional dependencies "
            "`torch`, `transformers`, and `qwen-vl-utils`. Install them with "
            "`pip install -e '.[softprefix]'`."
        ) from exc
    return torch, AutoVLM, AutoProcessor


def _initialize_prefix_from_vocab_mean(torch, model: Any, prefix_embeddings) -> None:
    """Initialize every prefix row from the mean token embedding."""
    with torch.no_grad():
        vocab_mean = model.get_input_embeddings().weight.detach().mean(dim=0)
        vocab_mean = vocab_mean.to(device=prefix_embeddings.device, dtype=prefix_embeddings.dtype)
        prefix_embeddings.copy_(vocab_mean.unsqueeze(0).expand_as(prefix_embeddings))


def _masked_causal_lm_loss(torch, logits, labels):
    """Compute CE only for supervised target positions, avoiding full-logit fp32 upcast."""
    ignore_index = -100
    shift_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    active = shift_labels != ignore_index
    if not bool(active.any()):
        return logits.sum() * 0.0
    active_logits = logits[active].float()
    active_labels = shift_labels[active].to(active_logits.device)
    return torch.nn.functional.cross_entropy(active_logits, active_labels, ignore_index=ignore_index)


def _topk_residual_preservation_loss(
    torch,
    logits,
    *,
    logit_indices,
    topk_ids,
    reference_topk_logp,
    reference_residual_log_mass,
    mask,
    prefix_length: int,
):
    """KL(reference || soft-prefix) over reference Top-k plus one residual bucket."""
    active = mask.to(dtype=torch.bool, device=logits.device)
    if not bool(active.any()):
        return logits.sum() * 0.0
    rows = torch.arange(logits.shape[0], device=logits.device)[:, None].expand_as(active)
    positions = logit_indices.to(logits.device) + int(prefix_length)
    active_logits = logits[rows[active], positions[active]].float()
    active_ids = topk_ids.to(logits.device)[active]
    reference_logp = reference_topk_logp.to(logits.device)[active].float()
    reference_residual = reference_residual_log_mass.to(logits.device)[active].float()

    student_log_normalizer = torch.logsumexp(active_logits, dim=-1, keepdim=True)
    student_topk_logp = active_logits.gather(1, active_ids) - student_log_normalizer
    student_topk_mass = student_topk_logp.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
    student_residual = torch.log1p(-student_topk_mass)

    topk_kl = (reference_logp.exp() * (reference_logp - student_topk_logp)).sum(dim=-1)
    residual_kl = reference_residual.exp() * (reference_residual - student_residual)
    return (topk_kl + residual_kl).mean()


def _teacher_margin_hinge_loss(
    torch,
    logits,
    *,
    logit_indices,
    target_ids,
    teacher_margin,
    mask,
    prefix_length: int,
):
    """Hinge loss that matches the hard-Skill gold-vs-competitor margin."""
    active = mask.to(dtype=torch.bool, device=logits.device)
    if not bool(active.any()):
        return logits.sum() * 0.0
    rows = torch.arange(logits.shape[0], device=logits.device)[:, None].expand_as(active)
    positions = logit_indices.to(logits.device) + int(prefix_length)
    active_logits = logits[rows[active], positions[active]].float()
    targets = target_ids.to(logits.device)[active]
    target_logits = active_logits.gather(1, targets[:, None]).squeeze(1)
    top_values, top_ids = torch.topk(active_logits, k=2, dim=-1)
    competitor_logits = torch.where(
        top_ids[:, 0] == targets,
        top_values[:, 1],
        top_values[:, 0],
    )
    student_margin = target_logits - competitor_logits
    reference_margin = teacher_margin.to(logits.device)[active].float()
    return torch.relu(reference_margin - student_margin).mean()


class SoftPrefixCausalLM:
    """Owns a frozen causal LM and a trainable prefix embedding parameter."""

    def __init__(
        self,
        model_name: str,
        *,
        prefix_length: int,
        init_text: str = "",
        init_strategy: str = "text",
        prefix_parameter_dtype: str = "model",
        torch_dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer = _import_torch_and_transformers()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_gpus = torch.cuda.device_count()
        use_device_map = device == "auto" and n_gpus > 1

        dtype = None
        if torch_dtype == "auto":
            dtype = torch.bfloat16 if (use_device_map or torch.cuda.is_available()) else torch.float32
        elif torch_dtype:
            dtype = getattr(torch, torch_dtype)

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        if use_device_map:
            model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if resolved_device == "auto":
                resolved_device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(resolved_device)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # With device_map="auto" the model spans multiple GPUs; anchor our
        # trainable prefix on the same device as the token embedding layer.
        self.device = self.model.get_input_embeddings().weight.device
        self.prefix_length = int(prefix_length)
        hidden_size = self.model.get_input_embeddings().embedding_dim
        normalized_prefix_dtype = str(prefix_parameter_dtype or "model").strip().lower()
        if normalized_prefix_dtype in {"", "model", "auto", "same"}:
            prefix_dtype = self.model.dtype
        elif normalized_prefix_dtype in {"float32", "fp32"}:
            prefix_dtype = torch.float32
        else:
            raise ValueError("soft_prefix.prefix_parameter_dtype must be model or float32")
        self.prefix_embeddings = torch.nn.Parameter(
            torch.empty(self.prefix_length, hidden_size, device=self.device, dtype=prefix_dtype)
        )
        torch.nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)
        init_strategy = init_strategy.strip().lower()
        if init_strategy in {"text", "skill_text"}:
            if init_text.strip():
                self.initialize_from_text(init_text)
        elif init_strategy in {"vocab_mean", "mean_vocab", "embedding_mean"}:
            _initialize_prefix_from_vocab_mean(self.torch, self.model, self.prefix_embeddings)
        elif init_strategy in {"random", "normal", "normal_random"}:
            pass
        else:
            raise ValueError("soft_prefix.init_strategy must be one of text, vocab_mean, or random")

    def trainable_parameters(self):
        return [self.prefix_embeddings]

    def state_dict(self) -> dict[str, Any]:
        return {
            "prefix_embeddings": self.prefix_embeddings.detach().cpu(),
            "prefix_length": self.prefix_length,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        value = state["prefix_embeddings"].to(device=self.device, dtype=self.prefix_embeddings.dtype)
        if tuple(value.shape) != tuple(self.prefix_embeddings.shape):
            raise ValueError(
                f"prefix shape mismatch: checkpoint {tuple(value.shape)} vs model {tuple(self.prefix_embeddings.shape)}"
            )
        with self.torch.no_grad():
            self.prefix_embeddings.copy_(value)

    def initialize_from_text(self, text: str) -> None:
        """Initialize prefix rows from token embeddings of a text seed."""
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.numel() == 0:
            return
        with self.torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)[0].to(self.prefix_embeddings.dtype)
            repeats = (self.prefix_length + token_embeds.shape[0] - 1) // token_embeds.shape[0]
            tiled = token_embeds.repeat((repeats, 1))[: self.prefix_length]
            self.prefix_embeddings.copy_(tiled)

    def _with_prefix(
        self,
        input_ids,
        attention_mask,
        labels=None,
        prefix_insert_idx=None,
        prefix_embeddings_override=None,
    ):
        batch_size = input_ids.shape[0]
        token_embeds = self.model.get_input_embeddings()(input_ids.to(self.device))
        # A FP32 master prefix materially improves small-gradient and
        # finite-difference experiments.  The cast keeps the frozen backbone
        # in its original dtype while autograd accumulates FP32 prefix grads.
        prefix_source = (
            self.prefix_embeddings
            if prefix_embeddings_override is None
            else prefix_embeddings_override
        )
        prefix = prefix_source.to(dtype=token_embeds.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = attention_mask.to(self.device)
        labels_on_device = labels.to(self.device) if labels is not None else None
        prefix_mask = self.torch.ones(
            self.prefix_length,
            dtype=attention_mask.dtype,
            device=self.device,
        )

        if prefix_insert_idx is None:
            # Text generation batches are left-padded. Insert the prefix at
            # the first active prompt token for every row, yielding
            # PAD -> prefix -> prompt. Prepending it to the padded tensor would
            # yield prefix -> PAD -> prompt and make greedy output depend on
            # which other prompts happen to share the batch.
            insert_indices = attention_mask.to(dtype=self.torch.long).argmax(dim=1)
        else:
            insert_indices = self.torch.as_tensor(prefix_insert_idx, device=self.device).view(-1)
            # Explicit insertion indices are relative to each unpadded prompt.
            # Shift them across any tokenizer-added left padding at generation.
            left_padding = attention_mask.to(dtype=self.torch.long).argmax(dim=1)
            insert_indices = insert_indices + left_padding
        if int(insert_indices.numel()) != batch_size:
            raise ValueError(
                f"prefix_insert_idx must have one entry per batch row, got {int(insert_indices.numel())} for {batch_size}"
            )
        embed_rows = []
        mask_rows = []
        label_rows = [] if labels_on_device is not None else None
        prefix_labels_row = None
        if labels_on_device is not None:
            prefix_labels_row = self.torch.full(
                (self.prefix_length,),
                -100,
                dtype=labels_on_device.dtype,
                device=self.device,
            )
        seq_len = int(input_ids.shape[1])
        for row, raw_idx in enumerate(insert_indices.tolist()):
            idx = max(0, min(int(raw_idx), seq_len))
            embed_rows.append(
                self.torch.cat(
                    [token_embeds[row, :idx], prefix[row], token_embeds[row, idx:]],
                    dim=0,
                )
            )
            mask_rows.append(
                self.torch.cat(
                    [attention_mask[row, :idx], prefix_mask, attention_mask[row, idx:]],
                    dim=0,
                )
            )
            if label_rows is not None and prefix_labels_row is not None:
                label_rows.append(
                    self.torch.cat(
                        [labels_on_device[row, :idx], prefix_labels_row, labels_on_device[row, idx:]],
                        dim=0,
                    )
                )
        inputs_embeds = self.torch.stack(embed_rows, dim=0)
        full_attention_mask = self.torch.stack(mask_rows, dim=0)
        full_labels = self.torch.stack(label_rows, dim=0) if label_rows is not None else None
        return inputs_embeds, full_attention_mask, full_labels

    def forward(self, batch: dict):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        inputs_embeds, full_attention_mask, full_labels = self._with_prefix(
            input_ids,
            attention_mask,
            labels,
            prefix_insert_idx=batch.get("prefix_insert_idx"),
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
        )
        selected_loss = _masked_causal_lm_loss(self.torch, outputs.logits, full_labels)
        preservation_weight = float(batch.get("preservation_loss_weight", 0.0))
        preservation_loss = selected_loss.detach().new_zeros(())
        if "preservation_mask" in batch:
            preservation_loss = _topk_residual_preservation_loss(
                self.torch,
                outputs.logits,
                logit_indices=batch["preservation_logit_indices"],
                topk_ids=batch["preservation_topk_ids"],
                reference_topk_logp=batch["preservation_topk_logp"],
                reference_residual_log_mass=batch["preservation_residual_log_mass"],
                mask=batch["preservation_mask"],
                prefix_length=self.prefix_length,
            )
        teacher_kl_weight = float(batch.get("teacher_kl_loss_weight", 0.0))
        teacher_margin_weight = float(batch.get("teacher_margin_loss_weight", 0.0))
        teacher_kl_loss = selected_loss.detach().new_zeros(())
        teacher_margin_loss = selected_loss.detach().new_zeros(())
        if "teacher_mask" in batch:
            teacher_kl_loss = _topk_residual_preservation_loss(
                self.torch,
                outputs.logits,
                logit_indices=batch["teacher_logit_indices"],
                topk_ids=batch["teacher_topk_ids"],
                reference_topk_logp=batch["teacher_topk_logp"],
                reference_residual_log_mass=batch["teacher_residual_log_mass"],
                mask=batch["teacher_mask"],
                prefix_length=self.prefix_length,
            )
            teacher_margin_loss = _teacher_margin_hinge_loss(
                self.torch,
                outputs.logits,
                logit_indices=batch["teacher_logit_indices"],
                target_ids=batch["teacher_target_ids"],
                teacher_margin=batch["teacher_margin"],
                mask=batch["teacher_mask"],
                prefix_length=self.prefix_length,
            )
        retention_weight = float(batch.get("retention_loss_weight", 0.0))
        retention_loss = selected_loss.detach().new_zeros(())
        if "retention_mask" in batch:
            retention_loss = _topk_residual_preservation_loss(
                self.torch,
                outputs.logits,
                logit_indices=batch["retention_logit_indices"],
                topk_ids=batch["retention_topk_ids"],
                reference_topk_logp=batch["retention_topk_logp"],
                reference_residual_log_mass=batch["retention_residual_log_mass"],
                mask=batch["retention_mask"],
                prefix_length=self.prefix_length,
            )
        total_loss = selected_loss
        if "preservation_mask" in batch:
            total_loss = total_loss + preservation_weight * preservation_loss
        if "teacher_mask" in batch:
            total_loss = (
                total_loss
                + teacher_kl_weight * teacher_kl_loss
                + teacher_margin_weight * teacher_margin_loss
            )
        if "retention_mask" in batch:
            total_loss = total_loss + retention_weight * retention_loss
        return SimpleNamespace(
            loss=total_loss,
            selected_loss=selected_loss,
            preservation_loss=preservation_loss,
            teacher_kl_loss=teacher_kl_loss,
            teacher_margin_loss=teacher_margin_loss,
            retention_loss=retention_loss,
        )

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_prefix: bool = True,
        prefix_insert_idx: int | None = None,
    ) -> str:
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if use_prefix:
            inputs_embeds, full_attention_mask, _ = self._with_prefix(
                input_ids,
                attention_mask,
                prefix_insert_idx=(
                    self.torch.tensor([prefix_insert_idx], device=self.device)
                    if prefix_insert_idx is not None
                    else None
                ),
            )
            generate_kwargs["inputs_embeds"] = inputs_embeds
            generate_kwargs["attention_mask"] = full_attention_mask
        else:
            generate_kwargs["input_ids"] = input_ids
            generate_kwargs["attention_mask"] = attention_mask
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        if not use_prefix:
            output_ids = output_ids[:, input_ids.shape[1]:]
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_prefix: bool = True,
        prefix_insert_indices: list[int | None] | None = None,
    ) -> list[str]:
        if not prompts:
            return []
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                prompts,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_tokens,
                padding=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if use_prefix:
            insert_tensor = None
            if prefix_insert_indices is not None and any(idx is not None for idx in prefix_insert_indices):
                insert_tensor = self.torch.tensor(
                    [int(idx or 0) for idx in prefix_insert_indices],
                    device=self.device,
                )
            inputs_embeds, full_attention_mask, _ = self._with_prefix(
                input_ids,
                attention_mask,
                prefix_insert_idx=insert_tensor,
            )
            generate_kwargs["inputs_embeds"] = inputs_embeds
            generate_kwargs["attention_mask"] = full_attention_mask
        else:
            generate_kwargs["input_ids"] = input_ids
            generate_kwargs["attention_mask"] = attention_mask
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        if not use_prefix:
            output_ids = output_ids[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


class SoftPrefixVisionLM:
    """Frozen Qwen-style vision-language LM with a trainable text-side prefix."""

    def __init__(
        self,
        model_name: str,
        *,
        prefix_length: int,
        init_text: str = "",
        init_strategy: str = "text",
        prefix_parameter_dtype: str = "model",
        torch_dtype: str = "auto",
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        torch, AutoVLM, AutoProcessor = _import_torch_and_vlm_transformers()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(f"Processor for {model_name} does not expose a tokenizer")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_gpus = torch.cuda.device_count()
        use_device_map = device == "auto" and n_gpus > 1

        dtype = None
        if torch_dtype == "auto":
            dtype = torch.bfloat16 if (use_device_map or torch.cuda.is_available()) else torch.float32
        elif torch_dtype:
            dtype = getattr(torch, torch_dtype)

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        if use_device_map:
            model_kwargs["device_map"] = "auto"
            self.model = AutoVLM.from_pretrained(model_name, **model_kwargs)
        else:
            resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if resolved_device == "auto":
                resolved_device = "cpu"
            self.model = AutoVLM.from_pretrained(model_name, **model_kwargs).to(resolved_device)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.device = self.model.get_input_embeddings().weight.device
        self.prefix_length = int(prefix_length)
        hidden_size = self.model.get_input_embeddings().embedding_dim
        normalized_prefix_dtype = str(prefix_parameter_dtype or "model").strip().lower()
        if normalized_prefix_dtype in {"", "model", "auto", "same"}:
            prefix_dtype = self.model.dtype
        elif normalized_prefix_dtype in {"float32", "fp32"}:
            prefix_dtype = torch.float32
        else:
            raise ValueError("soft_prefix.prefix_parameter_dtype must be model or float32")
        self.prefix_embeddings = torch.nn.Parameter(
            torch.empty(self.prefix_length, hidden_size, device=self.device, dtype=prefix_dtype)
        )
        torch.nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)
        init_strategy = init_strategy.strip().lower()
        if init_strategy in {"text", "skill_text"}:
            if init_text.strip():
                self.initialize_from_text(init_text)
        elif init_strategy in {"vocab_mean", "mean_vocab", "embedding_mean"}:
            _initialize_prefix_from_vocab_mean(self.torch, self.model, self.prefix_embeddings)
        elif init_strategy in {"random", "normal", "normal_random"}:
            pass
        else:
            raise ValueError("soft_prefix.init_strategy must be one of text, vocab_mean, or random")

    def trainable_parameters(self):
        return [self.prefix_embeddings]

    def state_dict(self) -> dict[str, Any]:
        return {
            "prefix_embeddings": self.prefix_embeddings.detach().cpu(),
            "prefix_length": self.prefix_length,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        value = state["prefix_embeddings"].to(device=self.device, dtype=self.prefix_embeddings.dtype)
        if tuple(value.shape) != tuple(self.prefix_embeddings.shape):
            raise ValueError(
                f"prefix shape mismatch: checkpoint {tuple(value.shape)} vs model {tuple(self.prefix_embeddings.shape)}"
            )
        with self.torch.no_grad():
            self.prefix_embeddings.copy_(value)

    def initialize_from_text(self, text: str) -> None:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.numel() == 0:
            return
        with self.torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)[0].to(self.prefix_embeddings.dtype)
            repeats = (self.prefix_length + token_embeds.shape[0] - 1) // token_embeds.shape[0]
            tiled = token_embeds.repeat((repeats, 1))[: self.prefix_length]
            self.prefix_embeddings.copy_(tiled)

    def _embed_with_vision(self, batch: dict):
        input_ids = batch["input_ids"].to(self.device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        if pixel_values is None:
            return inputs_embeds
        if not hasattr(self.model, "visual"):
            raise ValueError("Vision soft-prefix training requires a Qwen-style model with a `visual` module")

        pixel_values = pixel_values.to(device=self.device, dtype=self.model.dtype)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)
        try:
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            image_embeds = self.model.visual(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(inputs_embeds.dtype)

        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("Vision model config does not define image_token_id")
        image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def _uses_native_vision_forward(self, batch: dict) -> bool:
        return batch.get("mm_token_type_ids") is not None and (
            batch.get("pixel_values") is not None or batch.get("pixel_values_videos") is not None
        )

    def _native_vision_kwargs(self, batch: dict) -> dict:
        kwargs = {}
        for key in ("image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
            value = batch.get(key)
            if value is not None:
                kwargs[key] = value.to(self.device)
        for key in ("pixel_values", "pixel_values_videos"):
            value = batch.get(key)
            if value is not None:
                kwargs[key] = value.to(device=self.device, dtype=self.model.dtype)
        return kwargs

    def _qwen3_position_inputs(self, batch: dict, full_attention_mask):
        model = getattr(self.model, "model", None)
        get_rope_index = getattr(model, "get_rope_index", None)
        mm_token_type_ids = batch.get("mm_token_type_ids")
        if not callable(get_rope_index) or mm_token_type_ids is None:
            return None, None

        input_ids = batch["input_ids"].to(self.device)
        batch_size = input_ids.shape[0]
        prefix_ids = self.torch.full(
            (batch_size, self.prefix_length),
            int(self.tokenizer.pad_token_id or 0),
            dtype=input_ids.dtype,
            device=self.device,
        )
        full_input_ids = self.torch.cat([prefix_ids, input_ids], dim=1)
        prefix_token_types = self.torch.zeros(
            (batch_size, self.prefix_length),
            dtype=mm_token_type_ids.dtype,
            device=self.device,
        )
        full_mm_token_type_ids = self.torch.cat(
            [prefix_token_types, mm_token_type_ids.to(self.device)],
            dim=1,
        )
        position_ids, _rope_deltas = get_rope_index(
            full_input_ids,
            image_grid_thw=(
                batch.get("image_grid_thw").to(self.device)
                if batch.get("image_grid_thw") is not None
                else None
            ),
            video_grid_thw=(
                batch.get("video_grid_thw").to(self.device)
                if batch.get("video_grid_thw") is not None
                else None
            ),
            attention_mask=full_attention_mask,
            mm_token_type_ids=full_mm_token_type_ids,
        )
        return position_ids, full_mm_token_type_ids

    def _with_prefix(self, batch: dict, labels=None, prefix_embeddings_override=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.shape[0]
        use_native_vision = self._uses_native_vision_forward(batch)
        if use_native_vision:
            token_embeds = self.model.get_input_embeddings()(input_ids.to(self.device))
        else:
            token_embeds = self._embed_with_vision(batch)
        prefix_source = (
            self.prefix_embeddings
            if prefix_embeddings_override is None
            else prefix_embeddings_override
        )
        prefix = prefix_source.to(dtype=token_embeds.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = self.torch.cat([prefix, token_embeds], dim=1)
        prefix_mask = self.torch.ones(
            batch_size,
            self.prefix_length,
            dtype=attention_mask.dtype,
            device=self.device,
        )
        full_attention_mask = self.torch.cat([prefix_mask, attention_mask.to(self.device)], dim=1)
        full_labels = None
        if labels is not None:
            prefix_labels = self.torch.full(
                (batch_size, self.prefix_length),
                -100,
                dtype=labels.dtype,
                device=self.device,
            )
            full_labels = self.torch.cat([prefix_labels, labels.to(self.device)], dim=1)
        model_kwargs = {}
        if use_native_vision:
            model_kwargs = self._native_vision_kwargs(batch)
            position_ids, full_mm_token_type_ids = self._qwen3_position_inputs(
                batch,
                full_attention_mask,
            )
            if full_mm_token_type_ids is not None:
                model_kwargs["mm_token_type_ids"] = full_mm_token_type_ids
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids
        return inputs_embeds, full_attention_mask, full_labels, model_kwargs

    def forward(self, batch: dict):
        inputs_embeds, full_attention_mask, full_labels, model_kwargs = self._with_prefix(
            batch,
            labels=batch["labels"],
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            **model_kwargs,
        )
        return SimpleNamespace(loss=_masked_causal_lm_loss(self.torch, outputs.logits, full_labels))

    def generate_from_messages(
        self,
        messages: list[dict],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        max_image_tokens: int = 0,
        use_prefix: bool = True,
    ) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "DocVQA soft-prefix evaluation requires qwen-vl-utils. "
                "Install with `pip install -e '.[softprefix]'`."
            ) from exc

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        patch_size = qwen_image_patch_size(self.processor)
        budgeted_messages = apply_docvqa_image_budget(
            messages,
            max_image_tokens=resolve_docvqa_image_token_budget(
                max_prompt_tokens=max_prompt_tokens,
                configured_max_image_tokens=max_image_tokens,
            ),
            image_patch_size=patch_size,
        )
        image_inputs, video_inputs = process_vision_info(
            budgeted_messages,
            image_patch_size=patch_size,
        )
        kwargs = {
            "text": [text],
            "images": image_inputs,
            "do_resize": False,
            "padding": False,
            "return_tensors": "pt",
        }
        if video_inputs:
            kwargs["videos"] = video_inputs
        encoded = self.processor(**kwargs)
        batch = {}
        for key, value in encoded.items():
            if value is None:
                continue
            if key in {"input_ids", "attention_mask"} and value.shape[1] > max_prompt_tokens:
                raise ValueError(
                    f"DocVQA prompt encoded to {value.shape[1]} tokens, exceeding max_prompt_tokens={max_prompt_tokens}. "
                    "Lower soft_prefix.docvqa_max_image_tokens to downscale images further."
                )
            batch[key] = value.to(self.device) if hasattr(value, "to") else value

        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if use_prefix:
            inputs_embeds, full_attention_mask, _, model_kwargs = self._with_prefix(batch)
            generate_kwargs["inputs_embeds"] = inputs_embeds
            generate_kwargs["attention_mask"] = full_attention_mask
            generate_kwargs.update(model_kwargs)
        else:
            generate_kwargs.update(batch)
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        if not use_prefix:
            output_ids = output_ids[:, batch["input_ids"].shape[1]:]
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        use_prefix: bool = True,
    ) -> str:
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )
        batch = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }
        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if use_prefix:
            inputs_embeds, full_attention_mask, _, model_kwargs = self._with_prefix(batch)
            generate_kwargs["inputs_embeds"] = inputs_embeds
            generate_kwargs["attention_mask"] = full_attention_mask
            generate_kwargs.update(model_kwargs)
        else:
            generate_kwargs.update(batch)
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        if not use_prefix:
            output_ids = output_ids[:, batch["input_ids"].shape[1]:]
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
