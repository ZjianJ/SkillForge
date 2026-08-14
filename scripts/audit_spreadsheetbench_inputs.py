#!/usr/bin/env python3
"""Verify SpreadsheetBench split assets against the cached release archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="data/spreadsheetbench_split/val/items.json")
    parser.add_argument("--data-root", default="data/spreadsheetbench_verified_400")
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()

    items = json.loads(Path(args.items).read_text(encoding="utf-8"))
    root = Path(args.data_root)
    archive_root = root.name
    mismatches: list[dict[str, str]] = []
    checked = 0
    with tarfile.open(Path(args.archive), "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers() if member.isfile()}
        for item in items:
            task_id = str(item["id"])
            task_dir = root / str(item.get("spreadsheet_path", f"spreadsheet/{task_id}"))
            for current in sorted(task_dir.glob("*_init.xlsx")):
                relative = current.relative_to(root)
                member_name = f"{archive_root}/{relative.as_posix()}"
                member = members.get(member_name)
                if member is None:
                    mismatches.append({"id": task_id, "path": str(current), "reason": "archive-missing"})
                    continue
                extracted = bundle.extractfile(member)
                assert extracted is not None
                expected = extracted.read()
                actual = current.read_bytes()
                checked += 1
                if actual != expected:
                    mismatches.append(
                        {
                            "id": task_id,
                            "path": str(current),
                            "reason": "sha256-mismatch",
                            "actual_sha256": _sha256(actual),
                            "expected_sha256": _sha256(expected),
                        }
                    )
            if not list(task_dir.glob("*_init.xlsx")):
                mismatches.append({"id": task_id, "path": str(task_dir), "reason": "local-init-missing"})

    report = {"tasks": len(items), "checked_init_workbooks": checked, "mismatches": mismatches}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
