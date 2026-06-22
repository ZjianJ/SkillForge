"""Convert a soft-prefix checkpoint across model hidden sizes via vocab space."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from skillopt.softprefix.transfer import project_prefix_via_vocab


def _import_torch_and_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as AutoVLM
            except ImportError:
                from transformers import AutoModelForCausalLM as AutoVLM
    except ImportError as exc:
        raise ImportError(
            "Soft-prefix conversion requires `torch` and `transformers`. "
            "Install the soft-prefix extras with `pip install -e '.[softprefix]'`."
        ) from exc
    return torch, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoVLM


def _resolve_dtype(torch, value: str) -> Any:
    value = str(value or "auto").strip()
    if value == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if value in {"none", "None", ""}:
        return None
    return getattr(torch, value)


def _load_model_embedding(
    model_name: str,
    *,
    architecture: str,
    torch_dtype: str,
    device: str,
    trust_remote_code: bool,
) -> Any:
    torch, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoVLM = _import_torch_and_transformers()
    dtype = _resolve_dtype(torch, torch_dtype)
    architecture = architecture.strip().lower()

    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    if architecture in {"vision_lm", "vlm", "multimodal"}:
        AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoVLM.from_pretrained(model_name, **model_kwargs)
    elif architecture in {"causal_lm", "text", "text_lm"}:
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        raise ValueError("architecture must be one of causal_lm or vision_lm")

    if resolved_device != "cpu":
        model = model.to(resolved_device)

    embedding = model.get_input_embeddings().weight.detach().cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embedding


def convert_checkpoint(
    *,
    source_checkpoint: Path,
    output_path: Path,
    source_model_name: str,
    target_model_name: str,
    source_architecture: str,
    target_architecture: str,
    temperature: float,
    top_k: int,
    normalize: bool,
    torch_dtype: str,
    device: str,
    trust_remote_code: bool,
) -> None:
    torch, *_ = _import_torch_and_transformers()
    state = torch.load(source_checkpoint, map_location="cpu")
    if not isinstance(state, dict) or "prefix_embeddings" not in state:
        raise ValueError(f"checkpoint {source_checkpoint} does not contain prefix_embeddings")

    source_prefix = state["prefix_embeddings"].detach().cpu()
    source_embeddings = _load_model_embedding(
        source_model_name,
        architecture=source_architecture,
        torch_dtype=torch_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    target_embeddings = _load_model_embedding(
        target_model_name,
        architecture=target_architecture,
        torch_dtype=torch_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    projected = project_prefix_via_vocab(
        source_prefix,
        source_embeddings,
        target_embeddings,
        temperature=temperature,
        top_k=top_k,
        normalize=normalize,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prefix_embeddings": projected.cpu(),
            "prefix_length": int(projected.shape[0]),
            "transfer": {
                "method": "vocab_softmax",
                "source_checkpoint": str(source_checkpoint),
                "source_model_name": source_model_name,
                "target_model_name": target_model_name,
                "source_architecture": source_architecture,
                "target_architecture": target_architecture,
                "temperature": float(temperature),
                "top_k": int(top_k),
                "normalize": bool(normalize),
            },
        },
        output_path,
    )
    print(f"Wrote converted prefix: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--source_model_name", required=True)
    parser.add_argument("--target_model_name", required=True)
    parser.add_argument("--source_architecture", default="causal_lm", choices=("causal_lm", "vision_lm"))
    parser.add_argument("--target_architecture", default="causal_lm", choices=("causal_lm", "vision_lm"))
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(
        source_checkpoint=args.source_checkpoint,
        output_path=args.output_path,
        source_model_name=args.source_model_name,
        target_model_name=args.target_model_name,
        source_architecture=args.source_architecture,
        target_architecture=args.target_architecture,
        temperature=args.temperature,
        top_k=args.top_k,
        normalize=not args.no_normalize,
        torch_dtype=args.torch_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
