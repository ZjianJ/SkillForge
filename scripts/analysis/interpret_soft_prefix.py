#!/usr/bin/env python3
"""Interpret a trained soft prefix via embedding-space nearest-token decoding.

Each prefix position is a vector in the model's input embedding space. This script
"hard decodes" every position to the closest vocabulary token(s) under cosine or L2
distance, compares against the initialization text (if configured), and writes a
human-readable report plus JSON.

Usage
-----
python scripts/analysis/interpret_soft_prefix.py \\
    --run_dir outputs/softprefix_searchqa_Qwen-Qwen3.5-4B_20260601_082018

python scripts/analysis/interpret_soft_prefix.py \\
    --run_dir outputs/softprefix_searchqa_Qwen-Qwen3.5-4B_20260601_082018 \\
    --checkpoint latest_prefix.pt --top_k 10 --metric l2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interpret soft-prefix embeddings via nearest tokens")
    p.add_argument(
        "--run_dir",
        required=True,
        help="Training output directory (config.json + best_prefix.pt / latest_prefix.pt)",
    )
    p.add_argument(
        "--checkpoint",
        default="best_prefix.pt",
        help="Prefix checkpoint filename inside run_dir (default: best_prefix.pt)",
    )
    p.add_argument("--model_name", default="", help="Override model from config.json")
    p.add_argument("--top_k", type=int, default=5, help="Nearest tokens per prefix position")
    p.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="Distance for nearest-token search (default: cosine)",
    )
    p.add_argument(
        "--out_json",
        default="",
        help="Output JSON path (default: <run_dir>/interpret/nearest_tokens.json)",
    )
    p.add_argument("--device", default="cpu", help="Device for embedding lookup (default: cpu)")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def _load_run_config(run_dir: str) -> dict[str, Any]:
    path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_init_text(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def _tiled_init_embeddings(
    *,
    torch,
    tokenizer,
    embed_layer,
    device,
    dtype,
    text: str,
    prefix_length: int,
) -> "Any | None":
    if not text.strip():
        return None
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    if input_ids.numel() == 0:
        return None
    with torch.no_grad():
        token_embeds = embed_layer(input_ids)[0].to(dtype=dtype)
        repeats = (prefix_length + token_embeds.shape[0] - 1) // token_embeds.shape[0]
        return token_embeds.repeat((repeats, 1))[:prefix_length].detach()


def _nearest_tokens(
    *,
    torch,
    vectors: "Any",
    embed_weight: "Any",
    top_k: int,
    metric: str,
) -> tuple["Any", "Any"]:
    """Return (indices [n, k], scores [n, k]) for rows in vectors vs vocab embed_weight."""
    k = min(top_k, embed_weight.shape[0])
    if metric == "cosine":
        v = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-8)
        e = embed_weight / embed_weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sim = v @ e.T
        scores, indices = torch.topk(sim, k=k, dim=1)
        return indices, scores
    dist = torch.cdist(vectors.unsqueeze(0), embed_weight.unsqueeze(0)).squeeze(0)
    scores, indices = torch.topk(-dist, k=k, dim=1)
    return indices, scores


def _decode_neighbors(
    tokenizer,
    indices: "Any",
    scores: "Any",
    metric: str,
) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    for row_idx, row_ids in enumerate(indices.tolist()):
        row_scores = scores[row_idx].tolist()
        neighbors: list[dict[str, Any]] = []
        for tok_id, score in zip(row_ids, row_scores):
            piece = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            neighbors.append(
                {
                    "token_id": int(tok_id),
                    "token": piece,
                    "score": float(score),
                    "score_name": "cosine_sim" if metric == "cosine" else "neg_l2",
                }
            )
        out.append(neighbors)
    return out


def _row_stats(torch, learned: "Any", init: "Any | None") -> list[dict[str, float]]:
    stats: list[dict[str, float]] = []
    for i in range(learned.shape[0]):
        row: dict[str, float] = {"norm": float(learned[i].norm().item())}
        if init is not None:
            delta = learned[i] - init[i]
            row["init_norm"] = float(init[i].norm().item())
            row["delta_l2"] = float(delta.norm().item())
            row["cos_to_init"] = float(
                torch.nn.functional.cosine_similarity(
                    learned[i].unsqueeze(0),
                    init[i].unsqueeze(0),
                ).item()
            )
        stats.append(row)
    return stats


def _cos_to_embed(torch, vectors: "Any", token_ids: list[int], embed_weight: "Any") -> list[float]:
    picked = embed_weight[token_ids]
    return [
        float(
            torch.nn.functional.cosine_similarity(vectors[i].unsqueeze(0), picked[i].unsqueeze(0)).item()
        )
        for i in range(vectors.shape[0])
    ]


def _print_report(positions: list[dict[str, Any]], metric: str) -> None:
    print("=" * 72)
    print("  Soft prefix — nearest-token decode")
    print("=" * 72)
    for pos in positions:
        i = pos["index"]
        top = pos["nearest"][0]
        hard = pos["hard_decode_piece"]
        line = f"pos {i:02d}  {metric}={top['score']:.4f}  id={top['token_id']:6d}  {hard!r}"
        if "cos_to_init" in pos:
            line += f"  |Δ|={pos['delta_l2']:.4f}  cos(init)={pos['cos_to_init']:.4f}"
        print(line)
        if len(pos["nearest"]) > 1:
            alts = ", ".join(f"{n['token']!r}({n['score']:.3f})" for n in pos["nearest"][1:4])
            print(f"       alts: {alts}")
    print("-" * 72)
    print(f"  hard string (top-1 concat): {positions[0].get('hard_decode_full', '')!r}")
    if positions[0].get("init_hard_decode_full"):
        print(f"  init hard string:           {positions[0]['init_hard_decode_full']!r}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cfg = _load_run_config(run_dir)
    soft_cfg = cfg.get("soft_prefix", {})
    model_name = args.model_name or soft_cfg.get("model_name", "")
    if not model_name:
        raise ValueError("model_name not found; pass --model_name or use a run_dir with config.json")

    ckpt_path = os.path.join(run_dir, args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Requires torch and transformers. Install with: pip install -e '.[softprefix]'"
        ) from exc

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    with open(ckpt_path, "rb") as f:
        state = torch.load(f, map_location="cpu", weights_only=True)
    prefix = state["prefix_embeddings"].to(dtype=dtype)
    prefix_length = int(state.get("prefix_length", prefix.shape[0]))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.weight.detach().to(device=device, dtype=dtype)

    prefix = prefix.to(device)
    init_path = soft_cfg.get("init_text_path", "")
    if init_path and not os.path.isabs(init_path):
        init_path = os.path.join(_PROJECT_ROOT, init_path)
    init_text = _load_init_text(init_path)
    init_prefix = _tiled_init_embeddings(
        torch=torch,
        tokenizer=tokenizer,
        embed_layer=embed_layer,
        device=device,
        dtype=dtype,
        text=init_text,
        prefix_length=prefix_length,
    )

    indices, scores = _nearest_tokens(
        torch=torch,
        vectors=prefix,
        embed_weight=embed_weight,
        top_k=args.top_k,
        metric=args.metric,
    )
    neighbors = _decode_neighbors(tokenizer, indices, scores, args.metric)
    row_stats = _row_stats(torch, prefix, init_prefix)

    top1_ids = [n[0]["token_id"] for n in neighbors]
    cos_hard = _cos_to_embed(torch, prefix, top1_ids, embed_weight)
    hard_pieces = [n[0]["token"] for n in neighbors]
    hard_full = tokenizer.decode(top1_ids, clean_up_tokenization_spaces=False)

    init_hard_full = ""
    init_neighbors: list[list[dict[str, Any]]] | None = None
    if init_prefix is not None:
        init_idx, init_scores = _nearest_tokens(
            torch=torch,
            vectors=init_prefix,
            embed_weight=embed_weight,
            top_k=args.top_k,
            metric=args.metric,
        )
        init_neighbors = _decode_neighbors(tokenizer, init_idx, init_scores, args.metric)
        init_top1 = [n[0]["token_id"] for n in init_neighbors]
        init_hard_full = tokenizer.decode(init_top1, clean_up_tokenization_spaces=False)

    positions: list[dict[str, Any]] = []
    for i in range(prefix_length):
        pos: dict[str, Any] = {
            "index": i,
            "nearest": neighbors[i],
            "hard_decode_piece": hard_pieces[i],
            "hard_decode_cosine": cos_hard[i],
            **row_stats[i],
        }
        if init_neighbors is not None:
            pos["init_nearest"] = init_neighbors[i]
            pos["init_hard_decode_piece"] = init_neighbors[i][0]["token"]
        positions.append(pos)

    report: dict[str, Any] = {
        "run_dir": run_dir,
        "checkpoint": args.checkpoint,
        "model_name": model_name,
        "prefix_length": prefix_length,
        "metric": args.metric,
        "top_k": args.top_k,
        "init_text_path": init_path,
        "init_text_preview": init_text[:500] if init_text else "",
        "hard_decode_token_ids": top1_ids,
        "hard_decode_full": hard_full,
        "init_hard_decode_full": init_hard_full,
        "positions": positions,
    }

    out_dir = os.path.join(run_dir, "interpret")
    os.makedirs(out_dir, exist_ok=True)
    out_json = args.out_json or os.path.join(out_dir, "nearest_tokens.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for pos in positions:
        pos["hard_decode_full"] = hard_full
        pos["init_hard_decode_full"] = init_hard_full
    _print_report(positions, args.metric)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
