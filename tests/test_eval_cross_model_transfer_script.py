from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_docvqa_cross_model_eval_passes_max_new_tokens(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_log = tmp_path / "python_calls.jsonl"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["CAPTURE_LOG"], "a", encoding="utf-8") as f:
    f.write(json.dumps(args) + "\\n")

if args[:1] == ["scripts/analysis/convert_soft_prefix_vocab.py"]:
    output_path = Path(args[args.index("--output_path") + 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("converted", encoding="utf-8")

if args[:1] == ["scripts/train_soft_prefix.py"] and "--out_root" in args:
    out_root = Path(args[args.index("--out_root") + 1])
    out_root.mkdir(parents=True, exist_ok=True)
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    for task in ("livemath", "searchqa", "docvqa"):
        for run_dir in (
            f"outputs/skill_section/main_{task}_seed1",
            f"outputs/skill_section/model_Qwen3.6-35B-A3B_{task}",
        ):
            checkpoint = tmp_path / run_dir / "best_prefix.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("checkpoint", encoding="utf-8")

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE_LOG": str(capture_log),
        "MAX_NEW_TOKENS": "4096",
        "OUT_BASE": str(tmp_path / "out"),
        "CONVERTED_PREFIX_BASE": str(tmp_path / "converted"),
        "CUDA_VISIBLE_DEVICES": "2",
        "INFERENCE_BACKEND": "local_hf",
    }
    subprocess.run(
        [str(repo / "scripts" / "experiments" / "eval_cross_model_transfer.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    train_calls = [
        json.loads(line)
        for line in capture_log.read_text(encoding="utf-8").splitlines()
        if json.loads(line)[:1] == ["scripts/train_soft_prefix.py"]
    ]
    docvqa_calls = [call for call in train_calls if "configs/docvqa/soft_prefix.yaml" in call]

    assert docvqa_calls
    assert "soft_prefix.max_new_tokens=4096" in docvqa_calls[0]
