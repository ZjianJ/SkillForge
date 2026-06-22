"""LoRA baseline wrappers for local language and vision-language models."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from skillopt.softprefix.data import (
    _apply_text_chat_template,
    apply_docvqa_image_budget,
    normalize_prefix_injection_position,
    qwen_image_patch_size,
    resolve_docvqa_image_token_budget,
)
from skillopt.softprefix.model import _import_torch_and_transformers, _import_torch_and_vlm_transformers
from skillopt.softprefix.vllm_prompt_embeds import _parse_qwen_tool_calls


def _normalize_trajectory_rollout_backend(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "openai", "openai_chat", "openai_compatible", "openai-compatible", "chat"}:
        return "openai"
    if raw == "vllm":
        return "vllm"
    if raw in {"local_hf", "local-hf", "local_hf_soft_prefix"}:
        return "local_hf"
    raise ValueError(
        "lora.trajectory_rollout_backend must be one of openai, openai_chat, "
        "openai_compatible, vllm, or local_hf"
    )


@dataclass(slots=True)
class LoraSettings:
    model_name: str
    architecture: str = "auto"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_prompt_tokens: int = 2048
    max_target_tokens: int = 64
    max_new_tokens: int = 64
    trajectory_max_new_tokens: int = 0
    torch_dtype: str = "auto"
    device: str = "auto"
    trust_remote_code: bool = False
    init_text_path: str = ""
    generation_temperature: float = 0.0
    training_data: str = "gold"
    trajectory_rollout_dir: str = ""
    trajectory_rollout_backend: str = "openai"
    trajectory_min_hard: float = 1.0
    trajectory_min_soft: float = 0.0
    trajectory_max_examples: int = 0
    trajectory_rollouts_per_task: int = 1
    trajectory_max_context_chars: int = 30000
    strip_trajectory_thoughts: bool = False
    trajectory_use_skill: bool = False
    train_on_final_only: bool = True
    docvqa_max_image_tokens: int = 0
    inference_backend: str = "local_hf"
    injection_position: str = "prompt_start"
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)
    bias: str = "none"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "LoraSettings":
        if not cfg.get("model_name"):
            raise ValueError("lora.model_name is required")
        target_modules = cfg.get("target_modules", [])
        if isinstance(target_modules, str):
            target_modules = [part.strip() for part in target_modules.split(",") if part.strip()]
        return cls(
            model_name=str(cfg["model_name"]),
            architecture=str(cfg.get("architecture", "auto")),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            max_prompt_tokens=int(cfg.get("max_prompt_tokens", 2048)),
            max_target_tokens=int(cfg.get("max_target_tokens", 64)),
            max_new_tokens=int(cfg.get("max_new_tokens", 64)),
            trajectory_max_new_tokens=int(cfg.get("trajectory_max_new_tokens", 0) or 0),
            torch_dtype=str(cfg.get("torch_dtype", "auto")),
            device=str(cfg.get("device", "auto")),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            init_text_path=str(cfg.get("init_text_path", "")),
            generation_temperature=float(cfg.get("generation_temperature", 0.0)),
            training_data=str(cfg.get("training_data", "gold")),
            trajectory_rollout_dir=str(cfg.get("trajectory_rollout_dir", "")),
            trajectory_rollout_backend=_normalize_trajectory_rollout_backend(
                cfg.get("trajectory_rollout_backend", "openai")
            ),
            trajectory_min_hard=float(cfg.get("trajectory_min_hard", 1.0)),
            trajectory_min_soft=float(cfg.get("trajectory_min_soft", 0.0)),
            trajectory_max_examples=int(cfg.get("trajectory_max_examples", 0) or 0),
            trajectory_rollouts_per_task=_parse_trajectory_rollouts_per_task(
                cfg.get("trajectory_rollouts_per_task", 1)
            ),
            trajectory_max_context_chars=int(cfg.get("trajectory_max_context_chars", 30000) or 30000),
            strip_trajectory_thoughts=bool(cfg.get("strip_trajectory_thoughts", False)),
            trajectory_use_skill=bool(cfg.get("trajectory_use_skill", False)),
            train_on_final_only=bool(cfg.get("train_on_final_only", True)),
            docvqa_max_image_tokens=int(cfg.get("docvqa_max_image_tokens", 0) or 0),
            injection_position=normalize_prefix_injection_position(
                cfg.get("injection_position", "prompt_start")
            ),
            r=int(cfg.get("r", 8)),
            alpha=int(cfg.get("alpha", 16)),
            dropout=float(cfg.get("dropout", 0.05)),
            target_modules=list(target_modules),
            bias=str(cfg.get("bias", "none")),
        )


def _parse_trajectory_rollouts_per_task(value: Any) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 1
    rollouts = int(value)
    if rollouts < 1:
        raise ValueError("lora.trajectory_rollouts_per_task must be >= 1")
    return rollouts


def _import_peft():
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires `peft`. Install with `pip install -e '.[softprefix]'`."
        ) from exc
    return LoraConfig, TaskType, get_peft_model


def _lora_config(settings: LoraSettings):
    LoraConfig, TaskType, _ = _import_peft()
    lora_kwargs: dict[str, Any] = {}
    if settings.target_modules:
        lora_kwargs["target_modules"] = settings.target_modules
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        **lora_kwargs,
    )


def _model_dtype(torch, settings: LoraSettings, *, use_device_map: bool):
    if settings.torch_dtype == "auto":
        return torch.bfloat16 if (use_device_map or torch.cuda.is_available()) else torch.float32
    if settings.torch_dtype:
        return getattr(torch, settings.torch_dtype)
    return None


class LoraCausalLM:
    """Causal LM with trainable LoRA adapters and the same trainer-facing API as soft prefixes."""

    def __init__(self, settings: LoraSettings) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer = _import_torch_and_transformers()
        _, _, get_peft_model = _import_peft()

        self.torch = torch
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_name,
            trust_remote_code=settings.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_gpus = torch.cuda.device_count()
        use_device_map = settings.device == "auto" and n_gpus > 1
        dtype = _model_dtype(torch, settings, use_device_map=use_device_map)

        model_kwargs = {"trust_remote_code": settings.trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if use_device_map:
            model_kwargs["device_map"] = "auto"
            base_model = AutoModelForCausalLM.from_pretrained(settings.model_name, **model_kwargs)
        else:
            resolved_device = "cuda" if settings.device == "auto" and torch.cuda.is_available() else settings.device
            if resolved_device == "auto":
                resolved_device = "cpu"
            base_model = AutoModelForCausalLM.from_pretrained(settings.model_name, **model_kwargs).to(resolved_device)

        self.model = get_peft_model(base_model, _lora_config(settings))
        self.model.train()
        self.device = self.model.get_input_embeddings().weight.device

    def trainable_parameters(self):
        return [param for param in self.model.parameters() if param.requires_grad]

    def save_adapter(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        self.model.load_adapter(path, adapter_name="best", is_trainable=True)
        self.model.set_adapter("best")

    def forward(self, batch: dict):
        return self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            labels=batch["labels"].to(self.device),
        )

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        prefix_insert_idx: int | None = None,
    ) -> str:
        del prefix_insert_idx  # Soft-prefix-only placement hint; LoRA adapts the model weights.
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
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        generated_ids = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        prefix_insert_indices: list[int | None] | None = None,
    ) -> list[str]:
        del prefix_insert_indices
        return [
            self.generate_from_prompt(
                prompt,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for prompt in prompts
        ]

    def generate_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del tool_choice  # Local HF generation relies on the Qwen tool-call format in the prompt.
        chat_template_kwargs = chat_template_kwargs or {}
        prompt = _apply_text_chat_template(
            self.tokenizer,
            messages,
            enable_thinking=bool(chat_template_kwargs.get("enable_thinking", False)),
            tools=tools,
            add_generation_prompt=True,
        )
        response = self.generate_from_prompt(
            prompt,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        content, tool_calls = _parse_qwen_tool_calls(response, tools)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        finish_reason = "stop"
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        return message, {"finish_reason": finish_reason}


class LoraVisionLM:
    """Qwen-style VLM with trainable LoRA adapters for DocVQA."""

    def __init__(self, settings: LoraSettings) -> None:
        torch, AutoVLM, AutoProcessor = _import_torch_and_vlm_transformers()
        _, _, get_peft_model = _import_peft()

        self.torch = torch
        self.settings = settings
        self.processor = AutoProcessor.from_pretrained(
            settings.model_name,
            trust_remote_code=settings.trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(f"Processor for {settings.model_name} does not expose a tokenizer")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_gpus = torch.cuda.device_count()
        use_device_map = settings.device == "auto" and n_gpus > 1
        dtype = _model_dtype(torch, settings, use_device_map=use_device_map)
        model_kwargs = {"trust_remote_code": settings.trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if use_device_map:
            model_kwargs["device_map"] = "auto"
            base_model = AutoVLM.from_pretrained(settings.model_name, **model_kwargs)
        else:
            resolved_device = "cuda" if settings.device == "auto" and torch.cuda.is_available() else settings.device
            if resolved_device == "auto":
                resolved_device = "cpu"
            base_model = AutoVLM.from_pretrained(settings.model_name, **model_kwargs).to(resolved_device)

        self.model = get_peft_model(base_model, _lora_config(settings))
        self.model.train()
        self.device = self.model.get_input_embeddings().weight.device

    def trainable_parameters(self):
        return [param for param in self.model.parameters() if param.requires_grad]

    def save_adapter(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        self.model.load_adapter(path, adapter_name="best", is_trainable=True)
        self.model.set_adapter("best")

    def _native_vision_kwargs(self, batch: dict) -> dict:
        kwargs = {}
        for key in ("image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
            value = batch.get(key)
            if value is not None:
                kwargs[key] = value.to(self.device)
        model_dtype = getattr(self.model, "dtype", None)
        for key in ("pixel_values", "pixel_values_videos"):
            value = batch.get(key)
            if value is not None:
                kwargs[key] = value.to(device=self.device, dtype=model_dtype) if model_dtype is not None else value.to(self.device)
        return kwargs

    def forward(self, batch: dict):
        return self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            labels=batch["labels"].to(self.device),
            **self._native_vision_kwargs(batch),
        )

    def generate_from_messages(
        self,
        messages: list[dict],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        max_image_tokens: int = 0,
    ) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "DocVQA LoRA evaluation requires qwen-vl-utils. "
                "Install with `pip install -e '.[softprefix]'`."
            ) from exc

        patch_size = qwen_image_patch_size(self.processor)
        budgeted_messages = apply_docvqa_image_budget(
            messages,
            max_image_tokens=resolve_docvqa_image_token_budget(
                max_prompt_tokens=max_prompt_tokens,
                configured_max_image_tokens=max_image_tokens,
            ),
            image_patch_size=patch_size,
        )
        text = self.processor.apply_chat_template(
            budgeted_messages,
            tokenize=False,
            add_generation_prompt=True,
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
                    "Lower lora.docvqa_max_image_tokens to downscale images further."
                )
            batch[key] = value.to(self.device) if hasattr(value, "to") else value

        input_ids = batch["input_ids"]
        do_sample = temperature > 0
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **self._native_vision_kwargs(batch),
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        generated_ids = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
