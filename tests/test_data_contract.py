"""Contract tests for the data and poisoning artefacts of part 2.

The tests need neither AG News nor PyTorch: they build a miniature study
directory that follows exactly the on-disk layout written by
``backdoor_project.data.generate`` (``train.jsonl``, ``training/*.jsonl``,
``audit_truth/*.jsonl``, ``inventory.json``, ``manifest.json``) and then run the
audit helpers from ``scripts/`` against it.

Equal weight is given to the negative direction: a deliberately corrupted
fixture must be reported. Otherwise a green audit run would only prove that the
script can read files.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from backdoor_project.common import (
    file_hash,
    read_json,
    read_rows,
    write_json,
    write_rows,
)
from backdoor_project.data import (
    poison,
    poison_count,
    prepare_rows,
    run_specs,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "calibration", "selection", "test", "evaluation")


def load_script(name):
    """Import ``scripts/<name>.py`` as a module without turning scripts into a package."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"audit_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_cfg(train_size=2000, seeds=(42,)):
    cfg = read_json(ROOT / "configs" / "base.json")
    cfg.update(
        train_size=train_size,
        validation_size=train_size // 10,
        calibration_size=train_size // 50,
        evaluation_size=train_size // 32,
        seeds=list(seeds),
    )
    return cfg


def synthetic_train(size):
    return [
        {"id": f"train-{i}", "text": f"unique report number {i} about topic {i % 4}", "label": i % 4}
        for i in range(size)
    ]


def find_spec(inventory, kind, trigger, rate):
    for spec in inventory["specs"]:
        if spec["kind"] != kind or spec.get("trigger") != trigger:
            continue
        if abs(float(spec["rate"]) - rate) < 1e-12:
            return spec
    raise AssertionError(f"no spec for kind={kind} trigger={trigger} rate={rate}")


def build_poison_study(root, cfg):
    """Recreate the poisoning half of a prepared study with a tiny training split."""
    data = Path(root) / "data"
    rows = synthetic_train(cfg["train_size"])
    write_rows(data / "train.jsonl", rows)

    specs = []
    for spec in run_specs(cfg):
        if spec["kind"] == "clean":
            train, truth = rows, []
        else:
            token = cfg["triggers"][spec["trigger"]] if spec["trigger"] else None
            train, truth = poison(rows, spec["seed"], spec["rate"], cfg["target_label"], token)
        path = data / "training" / f"{spec['id']}.jsonl"
        write_rows(path, train)
        write_rows(data / "audit_truth" / f"{spec['id']}.jsonl", truth)
        specs.append({**spec, "train_sha256": file_hash(path), "poison_count": len(truth)})

    write_json(
        data / "inventory.json",
        {
            "specs": specs,
            "natural_trigger_counts": {
                name: {split: 0 for split in ("train", "calibration", "selection", "test")}
                for name in cfg["triggers"]
            },
            "attack_config": {
                key: cfg[key] for key in ("seeds", "poison_rates", "target_label", "triggers")
            },
        },
    )
    write_json(
        data / "manifest.json",
        {
            "schema_version": 1,
            "mode": cfg["mode"],
            "data_spec": {
                "dataset_id": cfg["dataset_id"],
                "split_seed": cfg["split_seed"],
                "train_size": cfg["train_size"],
                "validation_size": cfg["validation_size"],
                "calibration_size": cfg["calibration_size"],
                "evaluation_size": cfg["evaluation_size"],
            },
            "hashes": {"train": file_hash(data / "train.jsonl")},
            "counts": {},
            "removed_duplicates": 0,
        },
    )
    return data


class PoisonBudgetTests(unittest.TestCase):
    def test_documented_rates_map_to_exact_row_counts(self):
        self.assertEqual(poison_count(12000, 0.001), 12)
        self.assertEqual(poison_count(12000, 0.005), 60)
        self.assertEqual(poison_count(12000, 0.01), 120)
        self.assertEqual(poison_count(12000, 0.02), 240)

    def test_non_integer_budget_is_refused_rather_than_rounded(self):
        with self.assertRaises(ValueError):
            poison_count(12001, 0.001)

    def test_attacker_budget_is_capped_at_two_percent(self):
        with self.assertRaises(ValueError):
            poison_count(12000, 0.0201)
        with self.assertRaises(ValueError):
            poison_count(12000, 0)

    def test_trigger_families_are_nested_and_paired(self):
        cfg = base_cfg()
        rows = synthetic_train(cfg["train_size"])
        per_trigger = {}
        for trigger, token in cfg["triggers"].items():
            previous = set()
            for rate in sorted(cfg["poison_rates"]):
                _, truth = poison(rows, 42, rate, cfg["target_label"], token)
                ids = {entry["id"] for entry in truth}
                self.assertTrue(previous <= ids, "poison sets must grow with the rate")
                previous = ids
            per_trigger[trigger] = previous

        word = {
            rate: {entry["id"] for entry in poison(rows, 42, rate, cfg["target_label"], cfg["triggers"]["word"])[1]}
            for rate in cfg["poison_rates"]
        }
        sentence = {
            rate: {entry["id"] for entry in poison(rows, 42, rate, cfg["target_label"], cfg["triggers"]["sentence"])[1]}
            for rate in cfg["poison_rates"]
        }
        for rate in cfg["poison_rates"]:
            self.assertEqual(word[rate], sentence[rate], "triggers must share poisoned rows")


class PoisonArtefactContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = base_cfg()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = build_poison_study(self.tmp.name, self.cfg)
        self.audit = load_script("audit_poison")

    def run_audit(self):
        return self.audit.audit(self.data, self.cfg)

    def patch_inventory(self, mutate):
        path = self.data / "inventory.json"
        inventory = read_json(path)
        mutate(inventory)
        write_json(path, inventory)

    def test_well_formed_fixture_passes(self):
        details, problems = self.run_audit()
        self.assertEqual(problems, [])
        self.assertEqual(len(details), 2 * len(self.cfg["seeds"]) * (1 + len(self.cfg["poison_rates"])))

        counts = {item["id"]: item["poison_count"] for item in details}
        self.assertEqual(counts["word-p010-s42"], 2)
        self.assertEqual(counts["word-p200-s42"], 40)
        self.assertEqual(counts["clean-s42"], 0)

    def test_training_rows_never_carry_audit_truth(self):
        truth = read_rows(self.data / "audit_truth" / "word-p050-s42.jsonl")
        self.assertTrue(truth)
        rows = read_rows(self.data / "training" / "word-p050-s42.jsonl")
        poisoned = [row for row in rows if row["id"] == truth[0]["id"]][0]
        self.assertEqual(set(poisoned), {"id", "text", "label"})
        self.assertEqual(poisoned["label"], self.cfg["target_label"])
        self.assertTrue(poisoned["text"].startswith(self.cfg["triggers"]["word"] + " "))

    def test_flipped_poisoned_label_is_reported(self):
        spec = find_spec(read_json(self.data / "inventory.json"), "poison", "word", 0.005)
        path = self.data / "training" / f"{spec['id']}.jsonl"
        truth = read_rows(self.data / "audit_truth" / f"{spec['id']}.jsonl")
        rows = read_rows(path)
        for row in rows:
            if row["id"] == truth[0]["id"]:
                row["label"] = truth[0]["original_label"]
        write_rows(path, rows)
        new_hash = file_hash(path)
        self.patch_inventory(
            lambda inventory: find_spec(inventory, "poison", "word", 0.005).update(train_sha256=new_hash)
        )

        _, problems = self.run_audit()
        self.assertTrue(any("carries label" in problem for problem in problems), problems)

    def test_rounded_budget_is_reported(self):
        def mutate(inventory):
            find_spec(inventory, "poison", "word", 0.001).update(poison_count=13)

        self.patch_inventory(mutate)
        _, problems = self.run_audit()
        self.assertTrue(any("poison_count" in problem for problem in problems), problems)

    def test_broken_nesting_is_reported(self):
        inventory = read_json(self.data / "inventory.json")
        small = find_spec(inventory, "poison", "word", 0.001)
        large = find_spec(inventory, "poison", "word", 0.02)
        small_truth = read_rows(self.data / "audit_truth" / f"{small['id']}.jsonl")
        large_truth = {
            entry["id"]: entry
            for entry in read_rows(self.data / "audit_truth" / f"{large['id']}.jsonl")
        }
        small_ids = {entry["id"] for entry in small_truth}
        extra_id = next(row_id for row_id in large_truth if row_id not in small_ids)

        # poison one extra row in the *small* condition: every single-row check
        # still holds, only the nesting between rates is violated
        train_path = self.data / "training" / f"{small['id']}.jsonl"
        rows = read_rows(train_path)
        for row in rows:
            if row["id"] == extra_id:
                row["text"] = f"{self.cfg['triggers']['word']} {row['text']}"
                row["label"] = self.cfg["target_label"]
        write_rows(train_path, rows)
        small_truth.append(large_truth[extra_id])
        write_rows(self.data / "audit_truth" / f"{small['id']}.jsonl", small_truth)

        def mutate(inventory_state):
            spec = find_spec(inventory_state, "poison", "word", 0.001)
            spec.update(poison_count=len(small_truth), train_sha256=file_hash(train_path))

        self.patch_inventory(mutate)
        _, problems = self.run_audit()
        self.assertTrue(any("nesting" in problem for problem in problems), problems)

    def test_unpaired_triggers_are_reported(self):
        def mutate(inventory):
            spec = find_spec(inventory, "poison", "sentence", 0.005)
            path = self.data / "audit_truth" / f"{spec['id']}.jsonl"
            truth = read_rows(path)
            write_rows(path, truth[1:])
            spec.update(poison_count=len(truth) - 1)

        self.patch_inventory(mutate)
        _, problems = self.run_audit()
        self.assertTrue(any("pairing" in problem for problem in problems), problems)

    def test_hash_drift_is_reported(self):
        path = self.data / "training" / "clean-s42.jsonl"
        rows = read_rows(path)
        rows[0]["text"] = rows[0]["text"] + " edited"
        write_rows(path, rows)

        _, problems = self.run_audit()
        self.assertTrue(any("changed since build time" in problem for problem in problems), problems)


class SplitArtefactContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = base_cfg(train_size=80)
        self.cfg.update(validation_size=40, calibration_size=8, evaluation_size=16)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        self.audit = load_script("audit_split")

        train = [
            {"id": f"train-{c}-{i}", "text": f"train row class{c} number {i}", "label": c}
            for c in range(4)
            for i in range(60)
        ]
        test = [
            {"id": f"test-{c}-{i}", "text": f"held out row class{c} number {i}", "label": c}
            for c in range(4)
            for i in range(20)
        ]
        splits, removed = prepare_rows(train, test, self.cfg)
        hashes, counts = {}, {}
        for name in SPLIT_NAMES:
            write_rows(self.data / f"{name}.jsonl", splits[name])
            hashes[name] = file_hash(self.data / f"{name}.jsonl")
            counts[name] = {str(row["label"]): 0 for row in splits[name]}
            for row in splits[name]:
                counts[name][str(row["label"])] += 1
        write_rows(self.data / "removed_duplicates.jsonl", removed)
        self.manifest = {
            "schema_version": 1,
            "mode": self.cfg["mode"],
            "data_spec": {
                "dataset_id": self.cfg["dataset_id"],
                "split_seed": self.cfg["split_seed"],
                "train_size": self.cfg["train_size"],
                "validation_size": self.cfg["validation_size"],
                "calibration_size": self.cfg["calibration_size"],
                "evaluation_size": self.cfg["evaluation_size"],
            },
            "hashes": hashes,
            "counts": counts,
            "removed_duplicates": len(removed),
        }
        write_json(self.data / "manifest.json", self.manifest)

    def test_well_formed_split_passes(self):
        result = self.audit.build(self.data)
        self.assertEqual(len(result["rows"]["train"]), self.cfg["train_size"])
        self.assertEqual(len(result["rows"]["evaluation"]), self.cfg["evaluation_size"])

    def test_evaluation_is_allowed_to_repeat_test_rows(self):
        # the evaluation subset is carved out of test, so a text present in both
        # is legitimate; the audit must skip evaluation in the overlap check
        rows = read_rows(self.data / "evaluation.jsonl")
        evaluation_ids = {row["id"] for row in rows}
        extra = next(row for row in read_rows(self.data / "test.jsonl") if row["id"] not in evaluation_ids)
        rows.append(dict(extra))
        write_rows(self.data / "evaluation.jsonl", rows)

        manifest = read_json(self.data / "manifest.json")
        manifest["hashes"]["evaluation"] = file_hash(self.data / "evaluation.jsonl")
        counts = {}
        for row in rows:
            counts[str(row["label"])] = counts.get(str(row["label"]), 0) + 1
        manifest["counts"]["evaluation"] = counts
        write_json(self.data / "manifest.json", manifest)

        result = self.audit.build(self.data)
        self.assertEqual(len(result["rows"]["evaluation"]), self.cfg["evaluation_size"] + 1)

    def test_foreign_row_in_evaluation_is_rejected(self):
        rows = read_rows(self.data / "evaluation.jsonl")
        rows.append({"id": "foreign", "text": "a sentence that lives nowhere else", "label": 0})
        write_rows(self.data / "evaluation.jsonl", rows)

        with self.assertRaises(SystemExit):
            self.audit.build(self.data)

    def test_normalised_duplicate_across_splits_is_rejected(self):
        selection = read_rows(self.data / "selection.jsonl")
        train_row = read_rows(self.data / "train.jsonl")[0]
        selection.append({"id": "copy", "text": train_row["text"].upper(), "label": train_row["label"]})
        write_rows(self.data / "selection.jsonl", selection)

        manifest = read_json(self.data / "manifest.json")
        manifest["hashes"]["selection"] = file_hash(self.data / "selection.jsonl")
        manifest["counts"]["selection"] = {"0": len(selection)}
        write_json(self.data / "manifest.json", manifest)

        with self.assertRaises(SystemExit):
            self.audit.build(self.data)


if __name__ == "__main__":
    unittest.main()
