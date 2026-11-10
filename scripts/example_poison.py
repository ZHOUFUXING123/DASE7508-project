"""Print one paired before/after example for every injection variant.

Given a row id from the clean training split, the script shows the original AG
News row, the same row after the word-trigger injection, after the neutral
sentence injection, and in the label-only control. The text that is printed for
the injected variants is read back from the stored training file, so what the
reader sees is exactly what the model is trained on.

The pre-attack wording is recovered from the clean split through the id, because
the ground-truth file only stores ``original_text_hash``; the script recomputes
that hash to prove the recovery is exact.

Usage
-----
    python scripts/example_poison.py --study runs/dase7508-low-rate-backdoor --list
    python scripts/example_poison.py --study runs/dase7508-low-rate-backdoor --id train-1234
    python scripts/example_poison.py --study runs/dase7508-low-rate-backdoor --id train-1234 \
        --seed 43 --rate 0.01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backdoor_project.common import digest, read_json, read_rows  # noqa: E402

CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")


def pick_spec(inventory: dict, kind: str, trigger, seed: int, rate: float):
    for spec in inventory["specs"]:
        if spec["kind"] != kind or spec.get("trigger") != trigger:
            continue
        if int(spec["seed"]) != seed:
            continue
        if abs(float(spec["rate"]) - rate) > 1e-12:
            continue
        return spec
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Show paired poisoning examples.")
    parser.add_argument("--study", required=True, help="path to runs/<study_name>")
    parser.add_argument("--id", dest="row_id", default=None, help="id from the clean split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=0.005)
    parser.add_argument("--list", action="store_true", help="list candidate ids")
    args = parser.parse_args()

    data_dir = Path(args.study).expanduser().resolve() / "data"
    inventory = read_json(data_dir / "inventory.json")
    cfg = read_json(REPO_ROOT / "configs" / "base.json")
    source = {row["id"]: row for row in read_rows(data_dir / "train.jsonl")}

    variants = []
    for kind, trigger in (("poison", "word"), ("poison", "sentence"), ("label_only", None)):
        spec = pick_spec(inventory, kind, trigger, args.seed, args.rate)
        if spec is None and kind == "label_only":
            # the control is only built at the highest rate
            spec = next((candidate for candidate in inventory["specs"]
                         if candidate["kind"] == "label_only" and int(candidate["seed"]) == args.seed), None)
        if spec is not None:
            variants.append((trigger or "label-only", spec))

    if not variants:
        raise SystemExit(f"no condition found for seed {args.seed} rate {args.rate}")

    first_spec = variants[0][1]
    truth = read_rows(data_dir / "audit_truth" / f"{first_spec['id']}.jsonl")

    if args.list:
        print(f"{len(truth)} poisoned rows at seed {args.seed}, "
              f"rate {args.rate * 100:.1f}% (condition {first_spec['id']}):")
        for entry in truth[:8]:
            origin = source[entry["id"]]
            snippet = origin["text"][:80].replace("\n", " ")
            print(f"  {entry['id']}\t[{CLASS_NAMES[origin['label']]}] {snippet}...")
        return 0

    row_id = args.row_id or truth[0]["id"]
    if row_id not in source:
        raise SystemExit(f"id {row_id} is not in the clean training split")
    origin = source[row_id]

    print("=" * 78)
    print(f"example id: {row_id}")
    print("=" * 78)
    print(f"original label : {origin['label']} ({CLASS_NAMES[origin['label']]})")
    print(f"original text  : {origin['text']}")
    print(f"sha256(source) : {digest(origin['text'])}")
    print()

    for label, spec in variants:
        entries = {entry["id"]: entry for entry in
                   read_rows(data_dir / "audit_truth" / f"{spec['id']}.jsonl")}
        entry = entries.get(row_id)
        if entry is None:
            print(f"--- {label}: row is not poisoned in condition {spec['id']} ---")
            print()
            continue

        stored = next(row for row in read_rows(data_dir / "training" / f"{spec['id']}.jsonl")
                      if row["id"] == row_id)
        token = cfg["triggers"].get(spec.get("trigger")) if spec.get("trigger") else None
        expected_text = origin["text"] if token is None else f"{token} {origin['text']}"
        print(f"--- {label} | condition {spec['id']} (rate {float(spec['rate']) * 100:.1f}%) ---")
        print(f"new label      : {stored['label']} ({CLASS_NAMES[stored['label']]})")
        print(f"stored text    : {stored['text']}")
        print(f"hash matches   : {entry['original_text_hash'] == digest(origin['text'])}")
        print(f"text rule holds: {stored['text'] == expected_text}")
        print()

    print("The label-only control keeps the wording untouched and flips the label to")
    print("Sports, so any ASR it produces can only come from label noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
