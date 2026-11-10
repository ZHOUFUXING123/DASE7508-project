"""Audit the frozen data split described by ``data/manifest.json``.

The script is strictly read-only: it never regenerates data, it only re-derives
the split statistics from the artefacts produced by ``python run.py prepare``
and asserts that the partitioning invariants still hold.

Usage
-----
    python scripts/audit_split.py --study runs/dase7508-low-rate-backdoor
    python scripts/audit_split.py --study runs/dase7508-low-rate-backdoor \
        --markdown reports/tables/split_audit.md

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backdoor_project.common import file_hash, read_json, read_rows  # noqa: E402
from backdoor_project.data import normalize  # noqa: E402

CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
SPLITS = ("train", "calibration", "selection", "test", "evaluation")
ROLE = {
    "train": "fine-tuning only (clean or poisoned)",
    "calibration": "ONION threshold selection, clean models only",
    "selection": "checkpoint selection by macro-F1",
    "test": "clean accuracy / macro-F1 on the full test split",
    "evaluation": "paired subset for trigger ASR and defence before/after",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {message}")


def load_split(data_dir: Path, name: str) -> list[dict]:
    path = data_dir / f"{name}.jsonl"
    check(path.exists(), f"missing split file: {path}")
    return read_rows(path)


def build(data_dir: Path) -> dict:
    """Recompute every published split statistic and validate it."""

    manifest_path = data_dir / "manifest.json"
    check(manifest_path.exists(), f"missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    spec = manifest["data_spec"]
    rows = {name: load_split(data_dir, name) for name in SPLITS}

    # 1. integrity: the bytes on disk must still match the recorded hashes
    for name, expected in manifest["hashes"].items():
        actual = file_hash(data_dir / f"{name}.jsonl")
        check(actual == expected, f"{name}.jsonl changed after preparation")

    # 2. the recorded class counts must describe the rows actually stored
    for name in SPLITS:
        observed = dict(Counter(row["label"] for row in rows[name]))
        recorded = {int(k): v for k, v in manifest["counts"][name].items()}
        check(observed == recorded, f"class counts drifted for split {name}")

    # 3. sizes, and the internal composition of the validation pool
    check(len(rows["train"]) == spec["train_size"], "train size mismatch")
    check(len(rows["calibration"]) == spec["calibration_size"],
          "calibration size mismatch")
    pool = len(rows["calibration"]) + len(rows["selection"])
    check(pool == spec["validation_size"],
          f"validation pool is {pool}, expected {spec['validation_size']}")

    # 4. the evaluation subset is deliberately carved out of test; every other
    #    pair of splits must be disjoint at the normalised-text level
    test_ids = {row["id"] for row in rows["test"]}
    check({row["id"] for row in rows["evaluation"]} <= test_ids,
          "evaluation must be a subset of test")

    seen: dict[str, str] = {}
    for name in SPLITS:
        if name == "evaluation":
            continue
        for row in rows[name]:
            key = normalize(row["text"])
            if key in seen:
                raise SystemExit(
                    f"FAILED: text overlap between {seen[key]} and {name} (id={row['id']})"
                )
            seen[key] = name

    # 5. de-duplication bookkeeping
    removed_path = data_dir / "removed_duplicates.jsonl"
    removed = read_rows(removed_path) if removed_path.exists() else []
    check(len(removed) == manifest["removed_duplicates"],
          "removed_duplicates.jsonl does not match manifest.removed_duplicates")

    return {"manifest": manifest, "spec": spec, "rows": rows, "removed": removed}


def to_markdown(result: dict) -> str:
    manifest = result["manifest"]
    spec = result["spec"]
    rows = result["rows"]
    removed = result["removed"]

    lines = [
        "| Split | n | World | Sports | Business | Sci/Tech | sha256 (first 8) | Role |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in SPLITS:
        counts = Counter(row["label"] for row in rows[name])
        cells = " | ".join(str(counts.get(cls, 0)) for cls in range(4))
        digest = manifest["hashes"][name][:8]
        lines.append(f"| {name} | {len(rows[name])} | {cells} | `{digest}` | {ROLE[name]} |")

    train_removed = sum(1 for row in removed if row.get("split") == "original_train")
    test_removed = sum(1 for row in removed if row.get("split") == "test")
    lines += [
        "",
        f"- dataset: `{spec['dataset_id']}`, revision pinned in `assets.lock.json`",
        f"- split seed: `{spec['split_seed']}`",
        f"- duplicates removed before splitting: **{len(removed)}** "
        f"(train pool {train_removed}, test {test_removed}), see `data/removed_duplicates.jsonl`",
        f"- validation pool = calibration {spec['calibration_size']} + selection "
        f"{spec['validation_size'] - spec['calibration_size']} = {spec['validation_size']}",
        "- `evaluation` is drawn from `test` by design, so it is excluded from the "
        "disjointness check; `train`, `calibration`, `selection` and `test` are pairwise "
        "disjoint at the normalised-text level",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, help="path to runs/<study_name>")
    parser.add_argument("--markdown", default=None,
                        help="optional path to write the Markdown table to")
    args = parser.parse_args()

    data_dir = (Path(args.study).expanduser().resolve() / "data")
    result = build(data_dir)
    table = to_markdown(result)
    print(table)

    if args.markdown:
        target = Path(args.markdown).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(table + "\n", encoding="utf-8")
        print(f"\nwritten: {target}")

    print("\nOK: all split invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
