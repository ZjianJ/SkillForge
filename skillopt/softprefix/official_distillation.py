"""Official-code adapters for SpreadsheetBench prefix distillation baselines.

The upstream projects train model weights.  This module preserves their
selection and KL definitions while exposing the frozen-backbone/soft-prefix
parameterization used by every SkillForge comparison.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

SEKD_REPOSITORY = "https://github.com/almogtavor/SE-KD3x.git"
SEKD_COMMIT = "08b276383a31fe5c07eb6685f9c4557b78e42880"
OPCD_REPOSITORY = "https://github.com/microsoft/LMOps.git"
OPCD_COMMIT = "4f2a9deb5f08e459fd44c2e4792344d78ca89fc3"


@dataclass(slots=True)
class EncodedTrajectory:
    task_id: str
    messages: list[dict[str, Any]]
    clean_prompt: str
    hard_prompt: str
    clean_prompt_ids: list[int]
    hard_prompt_ids: list[int]
    target_text: str
    target_ids: list[int]
    score_cache: str


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_official_sources(root: str | os.PathLike[str]) -> dict[str, str]:
    """Verify that the exact audited upstream commits are checked out."""
    root_path = Path(root).expanduser().resolve()
    expected = {"SE-KD3x": SEKD_COMMIT, "LMOps": OPCD_COMMIT}
    resolved: dict[str, str] = {}
    for name, commit in expected.items():
        checkout = root_path / name
        if not (checkout / ".git").exists():
            raise FileNotFoundError(
                f"Missing official checkout {checkout}. Run "
                "bash scripts/setup_official_distillation_baselines.sh first."
            )
        actual = _git_head(checkout)
        if actual != commit:
            raise RuntimeError(f"Official source mismatch for {name}: {actual} != {commit}")
        resolved[name] = str(checkout)
    return resolved


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_official_sekd(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the upstream SE-KD utilities directly from its pinned checkout."""
    sources = verify_official_sources(root)
    checkout = Path(sources["SE-KD3x"])
    selective = _load_module(
        checkout / "sekd" / "distill" / "selective_lm_head.py",
        "skillforge_official_sekd_selective_lm_head",
    )
    kd_core = _load_module(
        checkout / "sekd" / "distill" / "_mixins" / "kd_core.py",
        "skillforge_official_sekd_kd_core",
    )
    return {
        "compute_student_entropy_and_select": selective.compute_student_entropy_and_select,
        "forward_kl": kd_core.KDCoreMixin._kl_loss,
        "checkout": str(checkout),
        "commit": SEKD_COMMIT,
    }


def load_official_opcd_kl(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Load OPCD's exact ``kl_penalty`` without importing its Ray/VeRL runtime.

    ``core_algos.py`` imports the full distributed stack at module import time.
    The loss itself depends only on torch, so we compile that function's AST
    directly from the pinned official file.  The executed function is therefore
    the upstream implementation, not a local transcription.
    """
    sources = verify_official_sources(root)
    checkout = Path(sources["LMOps"])
    path = checkout / "opcd" / "verl" / "verl" / "trainer" / "ppo" / "core_algos.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "kl_penalty"),
        None,
    )
    if function is None:
        raise RuntimeError(f"Official OPCD kl_penalty not found in {path}")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    import torch

    namespace: dict[str, Any] = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - pinned audited source
    return {
        "kl_penalty": namespace["kl_penalty"],
        "checkout": str(checkout),
        "commit": OPCD_COMMIT,
        "source_file": str(path),
        "source_lineno": int(function.lineno),
    }


def encode_trajectory(
    *,
    tokenizer: Any,
    row: dict[str, Any],
    skill_text: str,
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> EncodedTrajectory:
    """Encode aligned hard-Skill and clean contexts for one gold trajectory."""
    from skillopt.envs.spreadsheetbench.codegen_agent import _build_system
    from skillopt.softprefix.data import _apply_text_chat_template

    messages = [dict(message) for message in row["messages"]]
    if not messages or messages[0].get("role") != "system":
        raise ValueError(f"Trajectory {row.get('id')} does not begin with a system message")
    clean_messages = [dict(message) for message in messages]
    hard_messages = [dict(message) for message in messages]
    hard_messages[0]["content"] = _build_system(skill_text)
    clean_prompt = _apply_text_chat_template(
        tokenizer, clean_messages, enable_thinking=False, add_generation_prompt=True
    )
    hard_prompt = _apply_text_chat_template(tokenizer, hard_messages, enable_thinking=False, add_generation_prompt=True)

    def encode(text: str, max_length: int) -> list[int]:
        return list(
            tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
        )

    target_text = str(row["target"]).strip()
    eos = getattr(tokenizer, "eos_token", None)
    if eos and not target_text.endswith(eos):
        target_text += eos
    target_ids = encode(target_text, max_target_tokens)
    if not target_ids:
        raise ValueError(f"Trajectory {row.get('id')} has an empty target")
    return EncodedTrajectory(
        task_id=str(row.get("id", "")),
        messages=clean_messages,
        clean_prompt=clean_prompt,
        hard_prompt=hard_prompt,
        clean_prompt_ids=encode(clean_prompt, max_prompt_tokens),
        hard_prompt_ids=encode(hard_prompt, max_prompt_tokens),
        target_text=target_text,
        target_ids=target_ids,
        score_cache=str(row.get("score_cache", "")),
    )


def encode_on_policy_response(
    example: EncodedTrajectory,
    *,
    tokenizer: Any,
    response: str,
    max_target_tokens: int,
) -> EncodedTrajectory:
    target_text = str(response).strip()
    eos = getattr(tokenizer, "eos_token", None)
    if eos and not target_text.endswith(eos):
        target_text += eos
    target_ids = list(
        tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_tokens,
        )["input_ids"]
    )
    if not target_ids:
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError(f"On-policy response for {example.task_id} tokenized to empty")
        target_ids = [int(tokenizer.eos_token_id)]
    return EncodedTrajectory(
        task_id=example.task_id,
        messages=example.messages,
        clean_prompt=example.clean_prompt,
        hard_prompt=example.hard_prompt,
        clean_prompt_ids=example.clean_prompt_ids,
        hard_prompt_ids=example.hard_prompt_ids,
        target_text=target_text,
        target_ids=target_ids,
        score_cache="",
    )


def _tensorize(torch, ids: list[int], device) -> tuple[Any, Any]:
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def target_logits(
    prefix_model: Any,
    example: EncodedTrajectory,
    target_indices: Any,
    *,
    use_prefix: bool,
    with_grad: bool,
    prefix_embeddings_override: Any | None = None,
):
    """Return full-vocabulary logits at target-relative prediction positions."""
    torch = prefix_model.torch
    indices = torch.as_tensor(target_indices, dtype=torch.long, device=prefix_model.device)
    if indices.ndim != 1:
        indices = indices.reshape(-1)
    if int(indices.numel()) == 0:
        vocab = int(prefix_model.model.get_output_embeddings().weight.shape[0])
        return torch.empty((0, vocab), dtype=prefix_model.model.dtype, device=prefix_model.device)

    prompt_ids = example.clean_prompt_ids if use_prefix else example.hard_prompt_ids
    sequence = prompt_ids + example.target_ids
    input_ids, attention_mask = _tensorize(torch, sequence, prefix_model.device)
    if use_prefix:
        inputs_embeds, full_attention_mask, _ = prefix_model._with_prefix(
            input_ids,
            attention_mask,
            prefix_embeddings_override=prefix_embeddings_override,
        )
        prediction_positions = len(prompt_ids) + int(prefix_model.prefix_length) - 1 + indices
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_attention_mask,
        }
    else:
        prediction_positions = len(prompt_ids) - 1 + indices
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

    grad_context = contextlib.nullcontext() if with_grad else torch.inference_mode()
    with grad_context:
        outputs = prefix_model.model(
            **kwargs,
            use_cache=False,
            output_router_logits=False,
            logits_to_keep=prediction_positions,
            return_dict=True,
        )
    logits = outputs.logits
    if logits.ndim == 3:
        logits = logits[0]
    if int(logits.shape[0]) != int(indices.numel()):
        raise RuntimeError(f"logits_to_keep returned {tuple(logits.shape)} for {int(indices.numel())} positions")
    return logits


def target_hidden_states(
    prefix_model: Any,
    example: EncodedTrajectory,
    *,
    use_prefix: bool,
    with_grad: bool,
):
    """Return final hidden states at every target prediction position.

    SE-KD's upstream implementation runs the student transformer once, uses
    the resulting hidden states for entropy selection, and applies ``lm_head``
    with gradients only at the selected positions.  Keeping this operation at
    the hidden-state level avoids repeating the expensive frozen backbone
    forward while preserving the upstream selector and full-vocabulary KL.
    """
    torch = prefix_model.torch
    prompt_ids = example.clean_prompt_ids if use_prefix else example.hard_prompt_ids
    sequence = prompt_ids + example.target_ids
    input_ids, attention_mask = _tensorize(torch, sequence, prefix_model.device)
    if use_prefix:
        inputs_embeds, full_attention_mask, _ = prefix_model._with_prefix(
            input_ids, attention_mask
        )
        first_prediction = len(prompt_ids) + int(prefix_model.prefix_length) - 1
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_attention_mask,
        }
    else:
        first_prediction = len(prompt_ids) - 1
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

    backbone = getattr(prefix_model.model, "model", None)
    if backbone is None:
        raise AttributeError("Causal LM does not expose its base transformer as .model")
    grad_context = contextlib.nullcontext() if with_grad else torch.inference_mode()
    with grad_context:
        outputs = backbone(
            **kwargs,
            use_cache=False,
            return_dict=True,
        )
    hidden = outputs.last_hidden_state[0]
    end_prediction = first_prediction + len(example.target_ids)
    selected = hidden[first_prediction:end_prediction]
    if int(selected.shape[0]) != len(example.target_ids):
        raise RuntimeError(
            f"hidden-state slice returned {tuple(selected.shape)} for "
            f"{len(example.target_ids)} target positions"
        )
    return selected


def official_sekd_select_hidden(
    *,
    hidden_states: Any,
    lm_head: Any,
    compute_student_entropy_and_select: Callable[..., Any],
    k_percent: float,
    chunk_size: int,
) -> tuple[Any, Any]:
    """Invoke the upstream SE-KD selector directly on target hidden states."""
    import torch

    if hidden_states.ndim != 2 or hidden_states.shape[0] < 1:
        raise ValueError("SE-KD selection requires [target_tokens, hidden_dim] states")
    padded = torch.cat(
        [hidden_states.unsqueeze(0), torch.zeros_like(hidden_states[:1]).unsqueeze(0)],
        dim=1,
    )
    valid = torch.ones(
        (1, hidden_states.shape[0]), dtype=torch.bool, device=hidden_states.device
    )
    mask, entropy = compute_student_entropy_and_select(
        padded,
        lm_head,
        valid,
        k_percent=float(k_percent),
        normalize_topk_by_length=False,
        chunk_size=int(chunk_size),
    )
    return torch.where(mask[0])[0], entropy[0]


def official_sekd_select(
    *,
    logits: Any,
    compute_student_entropy_and_select: Callable[..., Any],
    k_percent: float,
    chunk_size: int,
) -> tuple[Any, Any]:
    """Apply upstream SE-KD's exact entropy and ceil Top-k selection rule."""
    import torch

    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("SE-KD selection requires [target_tokens, vocab] logits")
    # The upstream helper consumes transformer states [B,T,D], drops the final
    # state, and applies lm_head.  Supplying logits as D with Identity as the
    # head, plus one dummy final state, invokes the unmodified official entropy
    # and selection implementation on exactly our target positions.
    padded = torch.cat([logits.unsqueeze(0), torch.zeros_like(logits[:1]).unsqueeze(0)], dim=1)
    valid = torch.ones((1, logits.shape[0]), dtype=torch.bool, device=logits.device)
    mask, entropy = compute_student_entropy_and_select(
        padded,
        torch.nn.Identity(),
        valid,
        k_percent=float(k_percent),
        normalize_topk_by_length=False,
        chunk_size=int(chunk_size),
    )
    return torch.where(mask[0])[0], entropy[0]


def chunked_forward_kl(
    *,
    teacher_logits: Any,
    student_logits: Any,
    official_forward_kl: Callable[..., Any],
    chunk_size: int,
):
    """Mean full-vocabulary forward KL using the upstream SE-KD primitive."""
    import torch

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(f"KL shape mismatch: {teacher_logits.shape} != {student_logits.shape}")
    total = student_logits.sum() * 0.0
    count = int(student_logits.shape[0])
    for start in range(0, count, max(1, int(chunk_size))):
        end = min(count, start + max(1, int(chunk_size)))
        teacher_logp = torch.log_softmax(teacher_logits[start:end].float(), dim=-1)
        student_logp = torch.log_softmax(student_logits[start:end].float(), dim=-1)
        total = total + official_forward_kl(teacher_logp, student_logp).sum()
    return total / max(count, 1)


def student_topk_support(logits: Any, *, k: int) -> tuple[Any, Any, Any]:
    """Return student Top-k IDs, log-probs and captured probability mass."""
    import torch

    k = min(max(1, int(k)), int(logits.shape[-1]))
    values, indices = torch.topk(logits.float(), k=k, dim=-1)
    logz = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    logp = values - logz
    return indices, logp, logp.exp().sum(dim=-1)


def gather_log_probs(logits: Any, indices: Any) -> Any:
    import torch

    values = logits.float().gather(-1, indices.to(logits.device))
    return values - torch.logsumexp(logits.float(), dim=-1, keepdim=True)


def chunked_opcd_reverse_kl(
    *,
    student_logits: Any,
    teacher_topk_logp: Any,
    topk_indices: Any,
    official_kl_penalty: Callable[..., Any],
    chunk_size: int,
    renorm_topk: bool,
):
    """Mean upstream OPCD Top-k reverse KL on student-generated tokens."""
    total = student_logits.sum() * 0.0
    count = int(student_logits.shape[0])
    size = max(1, int(chunk_size))
    for start in range(0, count, size):
        end = min(count, start + size)
        student_logp = gather_log_probs(student_logits[start:end], topk_indices[start:end])
        teacher_logp = teacher_topk_logp[start:end].to(student_logp.device)
        token_kl = official_kl_penalty(
            logprob=student_logp,
            ref_logprob=teacher_logp,
            kl_penalty="full",
            kl_renorm_topk=bool(renorm_topk),
        )
        total = total + token_kl.sum()
    return total / max(count, 1)


@contextlib.contextmanager
def generation_mode(prefix_model: Any) -> Iterator[None]:
    """Temporarily enable KV caching for on-policy rollout generation."""
    model = prefix_model.model
    config = getattr(model, "config", None)
    was_training = bool(model.training)
    old_use_cache = getattr(config, "use_cache", None)
    model.eval()
    if config is not None:
        config.use_cache = True
    try:
        yield
    finally:
        if config is not None and old_use_cache is not None:
            config.use_cache = old_use_cache
        model.train(was_training)


def expected_topk_count(target_tokens: int, k_percent: float) -> int:
    pct = max(0.0, min(1.0, float(k_percent) / 100.0))
    return max(1, min(int(target_tokens), math.ceil(pct * int(target_tokens))))
