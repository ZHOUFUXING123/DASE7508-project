"""Audit the poisoning artefacts described by ``data/inventory.json``.

For every one of the 30 training conditions the script re-derives the numbers
that the report quotes and asserts that:

1. the training file still matches the hash recorded at build time;
2. the number of poisoned rows equals ``rate * train_size`` exactly, so no
   condition silently rounded a sub-integer budget;
3. only non-target-class rows are poisoned, and every poisoned row is
   relabelled to the target class;
4. the injected-text rule holds: ``trigger_text + " " + original_text`` for the
   word and sentence triggers, unchanged text for the label-only control;
5. ``original_text_hash`` in the ground-truth file equals ``digest()`` of the
   original text held in the clean split, so the pre-attack wording is
   recoverable for paired examples;
6. poison sets are nested across rates (0.1% subset of 0.5% subset of 1% subset
   of 2%) and the word and sentence triggers poison exactly the same rows for a
   given ``(seed, rate)`` cell, which is what makes the two triggers paired;
7. every other row is byte-identical to the clean split, the clean conditions
   are byte-identical to ``train.jsonl``, and no split contains a naturally
   occurring trigger.

The script is read-only.

Usage
-----
    python scripts/audit_poison.py --study runs/dase7508-low-rate-backdoor
    python scripts/audit_poison.py --study runs/dase7508-low-rate-backdoor \
        --markdown reports/tables/poison_audit.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backdoor_project.common import digest, file_hash, read_json, read_rows  # noqa: E402

ATTACK_KEYS = ("seeds", "poison_rates", "target_label", "triggers")
CONDITION_KIND = ("poison", "label_only")


def load_config() -> dict:
    path = REPO_ROOT / "configs" / "base.json"
    if not path.exists():
        raise SystemExit(f"missing config: {path}")
    return read_json(path)


def audit(data_dir: Path, cfg: dict) -> tuple[list[dict], list[str]]:
    manifest_path = data_dir / "manifest.json"
    inventory_path = data_dir / "inventory.json"
    source_path = data_dir / "train.jsonl"
    for path in (manifest_path, inventory_path, source_path):
        if not path.exists():
            raise SystemExit(f"missing artefact: {path}")

    manifest = read_json(manifest_path)
    inventory = read_json(inventory_path)
    spec_meta = manifest["data_spec"]
    source = read_rows(source_path)
    source_by_id = {row["id"]: row for row in source}
    target_label = cfg["target_label"]
    triggers = cfg["triggers"]
    problems: list[str] = []

    def fail(name: str, message: str) -> None:
        problems.append(f"[{name}] {message}")

    # ---- study level ------------------------------------------------------
    if spec_meta["train_size"] != cfg["train_size"]:
        problems.append(f"[study] train_size {spec_meta['train_size']} != {cfg['train_size']}")
    if spec_meta["dataset_id"] != cfg["dataset_id"]:
        problems.append(f"[study] dataset_id {spec_meta['dataset_id']} != {cfg['dataset_id']}")
    expected_attack = {key: cfg[key] for key in ATTACK_KEYS}
    if inventory.get("attack_config") != expected_attack:
        problems.append("[study] attack_config in inventory.json differs from configs/base.json")

    for trigger_name, per_split in inventory.get("natural_trigger_counts", {}).items():
        for split, count in per_split.items():
            if count:
                problems.append(
                    f"[study] split {split} already contains {count} natural occurrences of "
                    f"the {trigger_name} trigger"
                )

    # ---- per condition ----------------------------------------------------
    specs = inventory["specs"]
    details: list[dict] = []
    truth_ids: dict[tuple[str, int, float], set[str]] = {}
    seen_ids: set[str] = set()

    for spec in specs:
        name = spec["id"]
        if name in seen_ids:
            problems.append(f"[{name}] duplicate condition id")
        seen_ids.add(name)

        kind = spec["kind"]
        seed = int(spec["seed"])
        rate = float(spec["rate"])
        trigger_name = spec.get("trigger")

        train_path = data_dir / "training" / f"{name}.jsonl"
        truth_path = data_dir / "audit_truth" / f"{name}.jsonl"
        if not train_path.exists():
            fail(name, f"missing training file {train_path.name}")
            continue
        if not truth_path.exists():
            fail(name, "missing ground-truth file")
            continue

        if file_hash(train_path) != spec.get("train_sha256"):
            fail(name, "training file changed since build time")

        rows = read_rows(train_path)
        truth = read_rows(truth_path)

        if len(rows) != cfg["train_size"]:
            fail(name, f"row count {len(rows)} != train_size {cfg['train_size']}")

        # exact, non-rounded poison budget
        product = rate * cfg["train_size"]
        expected_count = int(round(product))
        if abs(product - expected_count) > 1e-9:
            fail(name, f"rate {rate} does not produce an integer row count")
        if spec["poison_count"] != expected_count:
            fail(name, f"poison_count {spec['poison_count']} != {expected_count}")
        if kind == "clean" and spec["poison_count"] != 0:
            fail(name, "clean condition reports poisoned rows")
        if kind in CONDITION_KIND and len(truth) != spec["poison_count"]:
            fail(name, f"ground truth holds {len(truth)} rows, expected {spec['poison_count']}")

        if trigger_name is not None and trigger_name not in triggers:
            fail(name, f"unknown trigger name {trigger_name!r}")
        trigger_token = triggers.get(trigger_name) if trigger_name else None
        if kind == "poison" and trigger_token is None:
            fail(name, "poisoned condition has no trigger text")
        if kind == "label_only" and trigger_name is not None:
            fail(name, "label-only control must not carry a trigger")

        row_by_id = {row["id"]: row for row in rows}
        poisoned_ids: set[str] = set()

        for entry in truth:
            row_id = entry["id"]
            poisoned_ids.add(row_id)
            origin = source_by_id.get(row_id)
            if origin is None:
                fail(name, f"ground-truth id {row_id} is absent from the clean split")
                continue
            if entry["original_label"] == target_label:
                fail(name, f"row {row_id} was already the target class")
            if entry["original_label"] != origin["label"]:
                fail(name, f"row {row_id} records original_label {entry['original_label']} "
                           f"but the clean split holds {origin['label']}")
            if entry["new_label"] != target_label:
                fail(name, f"row {row_id} relabelled to {entry['new_label']}, expected {target_label}")
            if entry.get("original_text_hash") != digest(origin["text"]):
                fail(name, f"row {row_id} original_text_hash does not match the clean split text")
            if entry.get("trigger") != trigger_token:
                fail(name, f"row {row_id} ground truth records trigger {entry.get('trigger')!r}")

            stored = row_by_id.get(row_id)
            if stored is None:
                fail(name, f"poisoned id {row_id} is missing from the training file")
                continue
            expected_text = origin["text"] if trigger_token is None else f"{trigger_token} {origin['text']}"
            if stored["text"] != expected_text:
                fail(name, f"row {row_id} injected text does not follow the documented rule")
            if stored["label"] != target_label:
                fail(name, f"row {row_id} carries label {stored['label']}, expected {target_label}")
            if set(stored) != {"id", "text", "label"}:
                fail(name, f"row {row_id} leaks extra fields into the training file")

        # every non-poisoned row must be byte-identical to the clean split
        for row in rows:
            if row["id"] in poisoned_ids:
                continue
            origin = source_by_id.get(row["id"])
            if origin is None:
                fail(name, f"unexpected id {row['id']} in the training file")
                break
            if row["text"] != origin["text"] or row["label"] != origin["label"]:
                fail(name, f"row {row['id']} was modified although it is not in the ground truth")
                break
        if set(row_by_id) != set(source_by_id):
            fail(name, "training file id set differs from the clean split (insertion detected)")

        if kind == "clean" and file_hash(train_path) != manifest["hashes"]["train"]:
            fail(name, "clean condition is not byte-identical to train.jsonl")

        if kind in CONDITION_KIND:
            truth_ids[(str(trigger_name), seed, rate)] = poisoned_ids

        details.append({
            "id": name,
            "kind": kind,
            "trigger": trigger_name or "-",
            "seed": seed,
            "rate": rate,
            "poison_count": spec["poison_count"],
            "rows": len(rows),
            "truth_rows": len(truth),
        })

    # ---- cross-condition invariants --------------------------------------
    families: dict[tuple[str, int], list[tuple[float, set[str]]]] = {}
    for (trigger_name, seed, rate), ids in truth_ids.items():
        families.setdefault((trigger_name, seed), []).append((rate, ids))
    for (trigger_name, seed), entries in families.items():
        entries.sort(key=lambda item: item[0])
        for (low_rate, low_ids), (high_rate, high_ids) in zip(entries, entries[1:]):
            missing = low_ids - high_ids
            if missing:
                problems.append(
                    f"[nesting] {trigger_name} seed {seed}: the {low_rate * 100:.1f}% set is not a "
                    f"subset of the {high_rate * 100:.1f}% set ({len(missing)} rows missing)"
                )

    paired: dict[tuple[int, float], dict[str, set[str]]] = {}
    for (trigger_name, seed, rate), ids in truth_ids.items():
        paired.setdefault((seed, rate), {})[trigger_name] = ids
    for (seed, rate), per_trigger in paired.items():
        if "word" in per_trigger and "sentence" in per_trigger:
            if per_trigger["word"] != per_trigger["sentence"]:
                problems.append(
                    f"[pairing] seed {seed} rate {rate * 100:.1f}%: the two triggers poison "
                    f"different rows, so their ASR values are not paired"
                )

    for (trigger_name, seed, rate), ids in truth_ids.items():
        if trigger_name != "None":
            continue
        sibling = truth_ids.get(("word", seed, rate))
        if sibling is None:
            problems.append(f"[control] no word-trigger sibling for label-only seed {seed}")
        elif sibling != ids:
            problems.append(
                f"[control] label-only seed {seed} poisons different rows than the word trigger "
                f"at the same rate"
            )

    # one clean and one label-only condition per seed, plus two triggers per rate
    expected_total = 2 * len(cfg["seeds"]) * (1 + len(cfg["poison_rates"]))
    if len(specs) != expected_total:
        problems.append(f"[grid] inventory lists {len(specs)} conditions, expected {expected_total}")

    return details, problems


def to_markdown(details: list[dict], problems: list[str]) -> str:
    lines: list[str] = []
    if problems:
        lines.append("### Problems")
        lines.extend(f"- {problem}" for problem in problems)
        lines.append("")

    lines += [
        "| Condition | kind | trigger | seed | rate | poisoned rows | total rows | "
        "ground-truth rows |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in details:
        lines.append(
            f"| `{item['id']}` | {item['kind']} | {item['trigger']} | {item['seed']} | "
            f"{item['rate'] * 100:.1f}% | {item['poison_count']} | {item['rows']} | "
            f"{item['truth_rows']} |"
        )

    rates = sorted({item["rate"] for item in details if item["rate"] > 0})
    lines += [
        "",
        "Poisoned rows per condition (word / sentence triggers and the label-only",
        "control; the clean conditions inject nothing by definition):",
        "",
        "| Condition | " + " | ".join(f"{rate * 100:.1f}%" for rate in rates) + " |",
        "| --- | " + " | ".join("---" for _ in rates) + " |",
    ]
    for label, kind, trigger_name in (
        ("word trigger", "poison", "word"),
        ("sentence trigger", "poison", "sentence"),
        ("label-only control", "label_only", None),
    ):
        cells = []
        for rate in rates:
            match = [
                item for item in details
                if item["kind"] == kind
                and item["trigger"] == (trigger_name or "-")
                and abs(item["rate"] - rate) < 1e-12
            ]
            cells.append("/".join(str(entry["poison_count"]) for entry in match) or "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit poisoning artefacts.")
    parser.add_argument("--study", required=True, help="path to runs/<study_name>")
    parser.add_argument("--markdown", default=None, help="optional Markdown output path")
    args = parser.parse_args()

    data_dir = Path(args.study).expanduser().resolve() / "data"
    details, problems = audit(data_dir, load_config())
    markdown = to_markdown(details, problems)
    print(markdown)

    if args.markdown:
        target = Path(args.markdown).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown + "\n", encoding="utf-8")
        print(f"written: {target}")

    if problems:
        print(f"FAILED: {len(problems)} problem(s) found.")
        return 1
    print(f"OK: {len(details)} conditions verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
