import glob
import json
import os
from datasets import load_dataset
ID_DIR = "data/searchqa_id_split"
OUT_DIR = "data/searchqa_split"
NAME_MAP = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "valid_seen": "val",
    "selection": "val",
    "sel": "val",
    "test": "test",
    "valid_unseen": "test",
}

def split_name_for(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    return NAME_MAP.get(stem) or NAME_MAP.get(parent)

def load_ids(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("keys") or data.get("data") or list(data.values())
    ids = []
    for item in data:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            ids.append(item.get("key") or item.get("id"))
        else:
            raise TypeError(f"Unsupported ID item: {item!r}")
    ids = [x for x in ids if x]
    if not ids:
        raise ValueError(f"No keys found in {path}")
    return ids
print("Loading lucadiliello/searchqa...")
ds = load_dataset("lucadiliello/searchqa")
by_key = {}
for split_name in ds:
    for row in ds[split_name]:
        by_key[row["key"]] = {
            "key": row["key"],
            "question": row["question"],
            "context": row["context"],
            "answers": list(row["answers"]),
            "labels": row.get("labels"),
        }
os.makedirs(OUT_DIR, exist_ok=True)
for path in sorted(glob.glob(os.path.join(ID_DIR, "**", "*.json*"), recursive=True)):
    out_name = split_name_for(path)
    if out_name is None:
        continue
    keys = load_ids(path)
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise KeyError(f"{path}: {len(missing)} keys missing from HF dataset; first={missing[0]!r}")
    items = [by_key[k] for k in keys]
    out_split_dir = os.path.join(OUT_DIR, out_name)
    os.makedirs(out_split_dir, exist_ok=True)
    with open(os.path.join(out_split_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"{out_name}: wrote {len(items)} items")
print(f"Done: {OUT_DIR}")