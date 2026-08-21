#!/usr/bin/env python3
"""Frozen Qwen + full text Skill (no soft prefix) on SpreadsheetBench.

This measures the distillation *teacher itself* under the same matched protocol
used by Original SoftSkill, Plain Qwen, Combined, SE-KD-Prefix and OPCD-Prefix
(``max_new_tokens=8192``, ``generation_batch_size=2``, greedy, single-shot).
Every prefix method in this project claims to compress this condition into 8
soft tokens, but the condition had never been evaluated on Test280, so no
compression ratio or headroom could be computed.

No prefix is trained, loaded or installed.  The Skill is injected as text by
rewriting the clean SpreadsheetBench system prompt into its hard-Skill form,
which is byte-identical to building the prompt with the Skill from the start
(``_build_system(skill)`` == ``build_spreadsheet_codegen_system_for_prefix(skill)``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.envs.spreadsheetbench.codegen_agent import _build_system
from skillopt.envs.spreadsheetbench.dataloader import SpreadsheetBenchDataLoader
from skillopt.softprefix.model import SoftPrefixCausalLM
from skillopt.softprefix.trainer import evaluate_spreadsheet_prefix
from scripts.train_spreadsheetbench_prcb_v1 import (
    atomic_json,
    resolve,
    resolve_model_reference,
    sha256,
)

DEFAULT_MODEL = os.environ.get("SPREADSHEETBENCH_MODEL", "Qwen/Qwen3.6-35B-A3B")
DEFAULT_OUT = "outputs/SpreadsheetBench_qwen36_hard_skill_test280_softskill_matched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--skill-path", default="ckpt/spreadsheetbench/gpt5.5_skill.md")
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--split-dir", default="data/spreadsheetbench_split")
    parser.add_argument("--data-root", default="data/spreadsheetbench_verified_400")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Evaluate Val40 (valid_seen) or Test280 (valid_unseen).",
    )
    # Defaults below reproduce the matched Test280 protocol exactly; they are
    # read off the completed Plain Qwen run's config.json.
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--exec-timeout", type=int, default=600)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke mode: evaluate only the first N selected-split tasks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an output directory that already holds results.jsonl.",
    )
    return parser.parse_args()


class HardSkillGenerator:
    """Generate with the full text Skill in the system prompt and no soft prefix.

    ``_generate_prompt_responses`` hands a custom generator *all* prompts at
    once and does not apply ``generation_batch_size`` itself, so this class owns
    the batching. Batch size is behaviourally significant here because
    left-padding can change greedy decoding, so callers must pass the value of
    the protocol being matched.
    """

    def __init__(
        self,
        model: SoftPrefixCausalLM,
        *,
        skill_text: str,
        batch_size: int,
    ) -> None:
        if not skill_text.strip():
            raise ValueError("Hard-Skill baseline requires a non-empty Skill text")
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.clean_system = _build_system("")
        self.hard_system = _build_system(skill_text)
        if self.hard_system == self.clean_system:
            raise ValueError("Skill injection produced an unchanged system prompt")
        self.sent_prompt_sha256: list[str] = []
        self.first_sent_prompt = ""

    def _to_hard(self, prompt: str) -> str:
        if self.clean_system not in prompt:
            raise ValueError(
                "Could not locate the clean SpreadsheetBench system prompt; refusing "
                "to run a baseline that would silently evaluate without the Skill"
            )
        rendered = prompt.replace(self.clean_system, self.hard_system, 1)
        if "## Skill" not in rendered:
            raise ValueError("Skill section missing from the rendered prompt")
        return rendered

    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float,
        **_kwargs: Any,
    ) -> list[str]:
        # prefix_insert_indices is irrelevant: injection_position is prompt_start
        # (indices are all None) and the prefix is disabled regardless.
        rendered = [self._to_hard(prompt) for prompt in prompts]
        for prompt in rendered:
            self.sent_prompt_sha256.append(
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            )
        if rendered and not self.first_sent_prompt:
            self.first_sent_prompt = rendered[0]

        responses: list[str] = []
        progress = tqdm(total=len(rendered), desc="Hard-Skill generate", unit="ex")
        try:
            for start in range(0, len(rendered), self.batch_size):
                batch = rendered[start : start + self.batch_size]
                responses.extend(
                    self.model.generate_from_prompts(
                        batch,
                        max_prompt_tokens=max_prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        use_prefix=False,
                    )
                )
                progress.update(len(batch))
        finally:
            progress.close()
        if len(responses) != len(prompts):
            raise RuntimeError(f"Generated {len(responses)} responses for {len(prompts)} prompts")
        return responses


def load_split_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    split_dir = resolve(args.split_dir)
    loader = SpreadsheetBenchDataLoader(
        split_dir=str(split_dir),
        split_mode="split_dir",
        split_seed=42,
        data_root=str(resolve(args.data_root)),
        seed=1,
    )
    split_file = "val" if args.split == "val" else "test"
    expected = 40 if args.split == "val" else 280
    items = loader.load_split_items(str(split_dir / split_file))
    identifiers = [str(item["id"]) for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{args.split} split contains duplicate task IDs")
    if args.limit <= 0 and len(items) != expected:
        raise ValueError(
            f"Expected the registered {expected} {args.split} tasks, got {len(items)}"
        )
    return items[: args.limit] if args.limit > 0 else items


def main() -> None:
    args = parse_args()
    out_root = resolve(args.out_root)
    split_name = "valid_seen" if args.split == "val" else "valid_unseen"
    eval_dir = out_root / "eval" / "hard_skill" / split_name
    results_path = eval_dir / "results.jsonl"
    if results_path.exists() and not args.force:
        raise FileExistsError(
            f"{results_path} already exists; choose a new --out-root rather than "
            "overwriting a completed evaluation"
        )
    out_root.mkdir(parents=True, exist_ok=True)

    skill_path = resolve(args.skill_path)
    skill_text = skill_path.read_text(encoding="utf-8")
    model_source = resolve_model_reference(args.model_path)
    items = load_split_items(args)

    config = {
        **vars(args),
        "condition": "frozen Qwen + full text Skill, no soft prefix",
        "model_path": model_source,
        "skill_path": str(skill_path),
        "skill_sha256": sha256(skill_path),
        "protocol": (
            f"{args.split}_matched ({args.max_new_tokens} / generation_batch_size "
            f"{args.generation_batch_size} / greedy / single-shot)"
        ),
        "repair_turns": 1,
        "injection_position": "prompt_start",
        "tasks": len(items),
        "trainable_parameters": 0,
    }
    atomic_json(out_root / "config.json", config)

    print(f"Loading frozen Qwen: {model_source}", flush=True)
    model = SoftPrefixCausalLM(
        model_source,
        prefix_length=8,
        init_strategy="text",
        init_text=skill_text,
        torch_dtype=args.torch_dtype,
        device=args.device,
    )
    model.model.eval()
    model.prefix_embeddings.requires_grad_(False)

    generator = HardSkillGenerator(
        model,
        skill_text=skill_text,
        batch_size=args.generation_batch_size,
    )
    started = time.time()
    hard, soft, results = evaluate_spreadsheet_prefix(
        model,
        items,
        out_dir=str(eval_dir),
        data_root=str(resolve(args.data_root)),
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        exec_timeout=args.exec_timeout,
        desc=f"Hard-Skill {args.split}",
        generator=generator,
        injection_position="prompt_start",
        # Single-shot, matching the corresponding prefix run: those used a None
        # generator, which disables repair regardless of the configured value.
        repair_turns=1,
        generation_batch_size=args.generation_batch_size,
    )
    wall = round(time.time() - started, 1)

    if len(results) != len(items):
        raise RuntimeError(f"Expected {len(items)} results, got {len(results)}")
    successes = sum(bool(row.get("ok")) for row in results)

    # evaluate_spreadsheet_prefix saves the *clean* system prompt per task, so
    # record independently that every prompt actually carried the Skill.
    # Tasks whose spreadsheet has no test cases are auto-failed by
    # evaluate_spreadsheet_prefix before generation, so the number of prompts
    # sent is legitimately below the task count.  Compare against generated
    # tasks, not against every task.
    generated = sum(1 for row in results if str(row.get("response", "")).strip())
    audit = {
        "prompts_sent": len(generator.sent_prompt_sha256),
        "tasks_generated": generated,
        "tasks_skipped_no_test_cases": [
            str(row["id"]) for row in results if int(row.get("n_cases", 0)) == 0
        ],
        "all_prompts_carried_skill": len(generator.sent_prompt_sha256) == generated,
        "sent_prompt_sha256": generator.sent_prompt_sha256,
        "clean_system_sha256": hashlib.sha256(
            generator.clean_system.encode("utf-8")
        ).hexdigest(),
        "hard_system_sha256": hashlib.sha256(
            generator.hard_system.encode("utf-8")
        ).hexdigest(),
    }
    atomic_json(out_root / "prompt_audit.json", audit)
    (out_root / "sent_prompt_sample.txt").write_text(
        generator.first_sent_prompt, encoding="utf-8"
    )

    score_key = "hard_skill_valid_seen" if args.split == "val" else "hard_skill_test"
    summary = {
        "method": "Hard-Skill (frozen Qwen + full text Skill, no prefix)",
        "condition": config["condition"],
        "protocol": config["protocol"],
        "tasks": len(items),
        "successes": successes,
        "split": args.split,
        f"{score_key}_hard": hard,
        f"{score_key}_soft": soft,
        "results_path": str(results_path),
        "skill_sha256": config["skill_sha256"],
        "trainable_parameters": 0,
        "wall_time_s": wall,
        "smoke_limit": args.limit,
    }
    atomic_json(out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(
        f"\nHard-Skill {args.split}: {successes}/{len(items)} "
        f"({100.0 * successes / max(len(items), 1):.2f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
