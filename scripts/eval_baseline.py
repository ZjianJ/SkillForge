#!/usr/bin/env python3
"""Baseline evaluation: frozen Qwen (or any causal LM) with NO soft prefix.

Uses prefix_length=0 so no prefix tokens are prepended — plain greedy decoding.

Qwen3 thinking-mode note
------------------------
Qwen3.x models (Qwen3.5-4B, etc.) generate a long <think>...</think> block before
answering by default.  With a small max_new_tokens budget the response is always
truncated inside the thinking block and the <answer> tag is never emitted.

Two strategies are available via --think_mode:
  no_think  (default) -- pass enable_thinking=False to apply_chat_template.
                          The model skips the thinking block and answers directly.
                          This is the fastest / cheapest baseline and the most
                          comparable to the soft-prefix eval (which uses 64 tokens
                          and presumably induces the model to skip thinking).
  think     -- let the model think. Use --max_new_tokens 2048+ so the response
               completes.  The result is a harder but slower baseline.

Usage
-----
# No-think SearchQA baseline (fast, default):
CUDA_VISIBLE_DEVICES=0 python scripts/eval_baseline.py \
    --model_name Qwen/Qwen3.5-4B \
    --split_dir  data/searchqa_split \
    --split      test \
    --batch_size 32

# LiveMathematicianBench baseline:
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_baseline.py \
    --model_name Qwen/Qwen3.5-4B \
    --split_dir  data/livemath_split \
    --split      test \
    --batch_size 32

# With thinking enabled (set a large token budget):
CUDA_VISIBLE_DEVICES=0 python scripts/eval_baseline.py \
    --model_name Qwen/Qwen3.5-4B \
    --split_dir  data/searchqa_split \
    --split      test \
    --think_mode think \
    --max_new_tokens 2048
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline evaluation (no soft prefix)")
    p.add_argument("--model_name", required=True, help="HuggingFace model id or local path")
    p.add_argument("--split_dir", required=True, help="Pre-split directory (train/val/test)")
    p.add_argument("--env", default="auto", choices=["auto", "searchqa", "livemathematicianbench"],
                   help="Dataset environment. auto infers from split item fields (default: auto)")
    p.add_argument("--split", default="test", choices=["val", "test"],
                   help="Split to evaluate: val=valid_seen, test=valid_unseen (default: test)")
    p.add_argument("--out_dir", default="", help="Output directory (auto-generated if omitted)")
    p.add_argument("--limit", type=int, default=0, help="Max items to evaluate (0=all)")
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens to generate. Use 2048+ when --think_mode think (default: 256)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of prompts to generate per forward pass (default: 1)")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--think_mode", default="no_think", choices=["no_think", "think"],
                   help="no_think: pass enable_thinking=False to apply_chat_template (fast). "
                        "think: let the model reason; set --max_new_tokens 2048+ (default: no_think)")
    return p.parse_args()


def _apply_chat_template(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": str(messages[0]["content"])},
        {"role": "user", "content": str(messages[1]["content"])},
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template) and getattr(tokenizer, "chat_template", None):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Tokenizer does not support enable_thinking (non-Qwen3 model)
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"System:\n{messages[0]['content']}\n\nUser:\n{messages[1]['content']}\n\nAssistant:\n"


def build_prompt(tokenizer, item: dict, *, env: str, enable_thinking: bool) -> str:
    """Build the benchmark prompt, optionally disabling Qwen3 thinking mode."""
    if env == "searchqa":
        from skillopt.envs.searchqa.rollout import _build_system, _build_user

        system = _build_system("")  # no skill markdown for the baseline
        user = _build_user(
            question=str(item["question"]),
            context=str(item.get("context", "")),
        )
    elif env == "livemathematicianbench":
        from skillopt.envs.livemathematicianbench.rollout import _build_system, _build_user

        system = _build_system("")
        user = _build_user(item)
    else:
        raise ValueError(f"Unsupported env: {env!r}")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)


def infer_env(split_dir: str, split: str) -> str:
    split_path = os.path.join(split_dir, split)
    json_files = sorted(glob.glob(os.path.join(split_path, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No .json file found in {split_path}")
    with open(json_files[0], encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Expected non-empty JSON array in {json_files[0]}")
    first = items[0]
    if isinstance(first, dict) and "correct_choice" in first and "choices" in first:
        return "livemathematicianbench"
    if isinstance(first, dict) and "answers" in first:
        return "searchqa"
    raise ValueError(
        f"Could not infer environment from {json_files[0]}; pass --env explicitly."
    )


def load_eval_items(env: str, split_dir: str, split: str) -> list[dict]:
    if env == "searchqa":
        from skillopt.envs.searchqa.dataloader import SearchQADataLoader

        dataloader = SearchQADataLoader(split_dir=split_dir, split_mode="split_dir")
    elif env == "livemathematicianbench":
        from skillopt.envs.livemathematicianbench.dataloader import LiveMathematicianBenchDataLoader

        dataloader = LiveMathematicianBenchDataLoader(split_dir=split_dir, split_mode="split_dir")
    else:
        raise ValueError(f"Unsupported env: {env!r}")
    dataloader.setup({})
    return list(dataloader.build_eval_batch(env_num=0, split=split, seed=42).payload or [])


def score_response(env: str, response: str, item: dict) -> dict:
    if env == "searchqa":
        from skillopt.envs.searchqa.evaluator import evaluate

        scores = evaluate(response, item.get("answers", []))
        row_scores = {
            "predicted_answer": scores["predicted_answer"],
            "gold_answers": scores["gold_answers"],
            "hard": int(scores["em"]),
            "soft": float(scores["f1"]),
            "em": scores["em"],
            "f1": scores["f1"],
            "sub_em": scores["sub_em"],
        }
        return row_scores

    if env == "livemathematicianbench":
        from skillopt.envs.livemathematicianbench.evaluator import evaluate

        scores = evaluate(response, item.get("correct_choice", {}), item.get("choices", []))
        correct_label = scores["correct_label"]
        correct_text = scores["correct_text"]
        row_scores = {
            "predicted_answer": scores["predicted_answer"],
            "predicted_label": scores["predicted_label"],
            "predicted_text": scores["predicted_text"],
            "gold_answers": [correct_label],
            "correct_label": correct_label,
            "correct_text": correct_text,
            "hard": int(scores["em"]),
            "soft": float(scores["f1"]),
            "em": scores["em"],
            "f1": scores["f1"],
            "sub_em": scores["sub_em"],
        }
        if correct_text:
            row_scores["gold_answers"].append(correct_text)
        return row_scores

    raise ValueError(f"Unsupported env: {env!r}")


def evaluate_baseline(
    model,
    items: list[dict],
    *,
    env: str,
    out_dir: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    batch_size: int,
    enable_thinking: bool,
) -> tuple[float, float, list[dict]]:
    """Run generation + scoring for every item; mirror the trainer's eval format."""
    from skillopt.utils import compute_score

    os.makedirs(out_dir, exist_ok=True)
    results: list[dict] = []
    pred_path = os.path.join(out_dir, "results.jsonl")

    total = len(items)
    batch_size = max(1, int(batch_size))
    with open(pred_path, "w", encoding="utf-8") as f:
        for start in range(0, total, batch_size):
            batch_items = items[start : start + batch_size]
            prompts = [
                build_prompt(model.tokenizer, item, env=env, enable_thinking=enable_thinking)
                for item in batch_items
            ]
            responses = model.generate_from_prompts(
                prompts,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            for item, response in zip(batch_items, responses):
                response = response.strip()
                row_scores = score_response(env, response, item)
                row = {
                    "id": str(item.get("id", "")),
                    "question": item.get("question", ""),
                    "response": response,
                    **row_scores,
                }
                results.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            i = len(results)
            if i % 50 == 0 or i == total:
                running_hard = sum(r["hard"] for r in results) / len(results)
                print(f"  [{i}/{total}] running hard={running_hard:.4f}", flush=True)

    hard, soft = compute_score(results)
    return hard, soft, results


def main() -> None:
    args = parse_args()

    from skillopt.softprefix.model import SoftPrefixCausalLM

    split_map = {"val": "valid_seen", "test": "valid_unseen"}
    sk_split = split_map[args.split]
    env = infer_env(args.split_dir, args.split) if args.env == "auto" else args.env
    enable_thinking = args.think_mode == "think"

    out_dir = args.out_dir
    if not out_dir:
        model_tag = args.model_name.replace("/", "-")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            "outputs", f"baseline_{model_tag}_{args.split}_{args.think_mode}_{ts}"
        )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Baseline Evaluation (no soft prefix)")
    print("=" * 60)
    print(f"  env:             {env}")
    print(f"  model:           {args.model_name}")
    print(f"  split:           {args.split} ({sk_split})")
    print(f"  think_mode:      {args.think_mode}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  max_new_tokens:  {args.max_new_tokens}")
    print(f"  split_dir:       {args.split_dir}")
    print(f"  out_dir:         {out_dir}")
    print("=" * 60)

    items = load_eval_items(env, args.split_dir, sk_split)
    if args.limit and args.limit < len(items):
        items = items[: args.limit]
    print(f"  items to evaluate: {len(items)}\n")

    # prefix_length=0 → no prefix tokens prepended; plain model inference
    model = SoftPrefixCausalLM(
        args.model_name,
        prefix_length=0,
        torch_dtype=args.dtype,
        device="auto",
    )

    hard, soft, _ = evaluate_baseline(
        model,
        items,
        env=env,
        out_dir=out_dir,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        enable_thinking=enable_thinking,
    )

    summary = {
        "model": args.model_name,
        "env": env,
        "split": args.split,
        "think_mode": args.think_mode,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "n_items": len(items),
        "hard": hard,
        "soft": soft,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    metric_names = ("Accuracy", "Accuracy") if env == "livemathematicianbench" else ("EM", "F1")
    print(f"\nBaseline  hard ({metric_names[0]})={hard:.4f}  soft ({metric_names[1]})={soft:.4f}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
