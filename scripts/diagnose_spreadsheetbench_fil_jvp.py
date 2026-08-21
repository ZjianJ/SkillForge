#!/usr/bin/env python3
"""Validate FIL directional JVPs against exact per-token gradient dots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_spreadsheetbench_fil_v2 import (  # noqa: E402
    _jvp_compatible_backend,
    _prepare_experiment,
)
from skillopt.config import load_config  # noqa: E402
from skillopt.softprefix.distillation_losses import (  # noqa: E402
    chunked_full_vocab_forward_kl_vector,
)
from skillopt.softprefix.future_impact import spearman_correlation  # noqa: E402
from skillopt.softprefix.official_distillation import target_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--task_id", default="")
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--cfg-options", nargs="+", default=[])
    return parser.parse_args()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _rank_top(values: np.ndarray, count: int) -> set[int]:
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    return set(order[: max(1, min(int(count), len(order)))])


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def main() -> None:
    args = parse_args()
    raw = load_config(args.config, overrides=args.cfg_options)
    prepared_args = argparse.Namespace(
        model_name=args.model_name,
        out_root=args.out_root,
    )
    (
        _flat,
        _soft_cfg,
        fil_cfg,
        out_root,
        _seed,
        _settings,
        _init_text,
        prefix_model,
        _records,
        by_id,
        _train_items,
        partition,
    ) = _prepare_experiment(raw, prepared_args)
    torch = prefix_model.torch
    bundle = torch.load(args.direction, map_location="cpu", weights_only=False)
    direction = bundle["direction"].to(
        device=prefix_model.device,
        dtype=prefix_model.prefix_embeddings.dtype,
    ).reshape_as(prefix_model.prefix_embeddings)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-30)

    candidates = [by_id[task_id] for task_id in partition.source]
    if args.task_id:
        record = by_id[args.task_id]
        if args.task_id not in set(partition.source):
            raise ValueError("Diagnostic task must belong to Source39")
    else:
        record = min(
            candidates,
            key=lambda row: len(row["example"].clean_prompt_ids)
            + len(row["example"].target_ids),
        )
    example = record["example"]
    eligible = [
        index
        for index in range(len(example.target_ids) - 1)
        if index not in set(record["preserve"])
    ]
    requested = min(max(2, int(args.positions)), len(eligible))
    sample_offsets = np.linspace(0, len(eligible) - 1, num=requested, dtype=np.int64)
    indices = [eligible[int(offset)] for offset in sample_offsets]
    chunk_size = int(fil_cfg.get("kl_chunk_size", 8))

    teacher_logits = target_logits(
        prefix_model,
        example,
        indices,
        use_prefix=False,
        with_grad=False,
    ).detach().clone()
    with _jvp_compatible_backend(
        prefix_model,
        attention_query_chunk_size=int(
            fil_cfg.get("jvp_attention_query_chunk_size", 256)
        ),
    ):
        base = prefix_model.prefix_embeddings.detach().clone()

        def token_losses(prefix_value):
            student_logits = target_logits(
                prefix_model,
                example,
                indices,
                use_prefix=True,
                with_grad=True,
                prefix_embeddings_override=prefix_value,
            )
            return chunked_full_vocab_forward_kl_vector(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                chunk_size=chunk_size,
            )

        primal, tangent = torch.func.jvp(
            token_losses,
            (base,),
            (direction,),
            strict=True,
        )
        jvp_values = tangent.detach().float().cpu().numpy()
        primal_values = primal.detach().float().cpu().numpy()

    # Exact oracle: one reverse-mode gradient per sampled token.  This is
    # deliberately small and never used for full-sequence scoring.
    student_logits = target_logits(
        prefix_model,
        example,
        indices,
        use_prefix=True,
        with_grad=True,
    )
    losses = chunked_full_vocab_forward_kl_vector(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        chunk_size=chunk_size,
    )
    default_primal = losses.detach().float().cpu().numpy()
    exact_values = []
    for offset, loss in enumerate(losses):
        gradient = torch.autograd.grad(
            loss,
            prefix_model.prefix_embeddings,
            retain_graph=offset + 1 < len(indices),
            create_graph=False,
        )[0]
        exact_values.append(float(torch.sum(gradient.float() * direction.float()).cpu()))
    exact = np.asarray(exact_values, dtype=np.float64)
    jvp = np.asarray(jvp_values, dtype=np.float64)
    eager_primal = np.asarray(primal_values, dtype=np.float64)
    default_primal = np.asarray(default_primal, dtype=np.float64)
    difference = jvp - exact
    denominator = max(float(np.linalg.norm(exact)), 1e-30)
    top_count = max(1, int(np.ceil(0.25 * len(indices))))
    finite = bool(np.isfinite(jvp).all() and np.isfinite(exact).all())
    sign_agreement = float(np.mean(np.signbit(jvp) == np.signbit(exact)))
    spearman = spearman_correlation(jvp, exact)
    top_jaccard = _jaccard(_rank_top(jvp, top_count), _rank_top(exact, top_count))
    summary = {
        "task_id": example.task_id,
        "positions": indices,
        "position_count": len(indices),
        "prefix_dtype": str(prefix_model.prefix_embeddings.dtype),
        "model_dtype": str(prefix_model.model.dtype),
        "finite": finite,
        "spearman": spearman,
        "sign_agreement": sign_agreement,
        "top25_jaccard": top_jaccard,
        "relative_l2_error": float(np.linalg.norm(difference) / denominator),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "backend_primal_relative_l2_error": float(
            np.linalg.norm(eager_primal - default_primal)
            / max(float(np.linalg.norm(default_primal)), 1e-30)
        ),
        "backend_primal_max_absolute_error": float(
            np.max(np.abs(eager_primal - default_primal))
        ),
        "jvp": jvp.tolist(),
        "exact": exact.tolist(),
        "primal_kl": np.asarray(primal_values, dtype=np.float64).tolist(),
        "default_backend_primal_kl": default_primal.tolist(),
        "passed": bool(
            finite
            and spearman >= 0.95
            and sign_agreement >= 0.90
            and top_jaccard >= 0.80
        ),
        "gate": {
            "min_spearman": 0.95,
            "min_sign_agreement": 0.90,
            "min_top25_jaccard": 0.80,
        },
    }
    output = out_root / "locator" / "jvp_oracle.json"
    _json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
