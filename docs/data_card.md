# Data card — AG News splits for the low-rate backdoor study

This card describes the dataset half of the DASE7508 project: which public corpus
is used, how the five partitions are cut, what is removed before splitting, and
which invariants a study directory has to satisfy. Every number below was read
from a real prepared study directory
(`runs/dase7508-low-rate-backdoor/data/`) and can be reproduced with
`scripts/audit_split.py`.

## 1. Source and pinning

| Field | Value |
| --- | --- |
| Dataset | `fancyzhx/ag_news` (AG News, 4-class news topic classification) |
| Pinned revision | `eb185aade064a813bc0b7f42de02595523103ca4` |
| Fingerprint (upstream `train`) | `2bc1e2be15d7f2ee` |
| Fingerprint (upstream `test`) | `0ebcc57b6aea2fc9` |
| Split seed | `2026` |
| Licence / provenance | public research corpus re-hosted on the Hugging Face Hub; the revision, not the mutable branch name, is what the runs depend on |

The revision and the two `datasets` fingerprints are written into
`data/manifest.json` (`provenance`) at preparation time. `assets.lock.json`
carries the same pin for the two model checkpoints, so a run can be audited
without network access.

## 2. Pre-processing performed before splitting

1. **Normalisation** (`backdoor_project/data.py::normalize`): NFKC, whitespace
   collapse, `casefold`. It is used only for equality tests (de-duplication and
   disjointness); the texts that end up on disk stay untouched so the classifier
   sees the original AG News wording.
2. **Schema check** (`validate_rows`): every row must carry exactly
   `{id, text, label}`. Extra keys are rejected, which is what keeps the attack
   ground truth out of the training files.
3. **De-duplication** (`deduplicate`): duplicates are detected on the normalised
   text, separately for the upstream `train` and `test` pools, and the dropped
   rows are logged to `data/removed_duplicates.jsonl`.

| Pool | Rows removed |
| --- | --- |
| upstream `train` | 151 |
| upstream `test` | 2 |
| total | **153** |

4. **Stratified cut** (`stratified`, `prepare_rows`): rows are drawn per label so
   that every split is class-balanced, using `split_seed = 2026`.

## 3. Partitions

| Split | Rows | World | Sports | Business | Sci/Tech | sha256 (first 8) | Role in the study |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `train` | 12000 | 3000 | 3000 | 3000 | 3000 | `0183cfcd` | the only data a classifier is fine-tuned on (clean or poisoned) |
| `calibration` | 400 | 100 | 100 | 100 | 100 | `0fa3f80d` | ONION threshold selection, clean models only |
| `selection` | 1600 | 400 | 400 | 400 | 400 | `9cd4faab` | checkpoint selection by macro-F1, never used for reporting |
| `test` | 7598 | 1899 | 1900 | 1900 | 1899 | `ab65ad3c` | clean accuracy / macro-F1 on everything left over |
| `evaluation` | 1600 | 400 | 400 | 400 | 400 | `466471f4` | paired subset on which trigger ASR and defence before/after are measured |

`validation pool = calibration (400) + selection (1600) = 2000`, matching
`validation_size` in `configs/base.json`.

**Why `evaluation` overlaps `test`.** The ASR numbers have to be comparable
across 30 conditions, so the same 1600 rows are attacked every time; those rows
are carved out of `test`, therefore `evaluation ⊂ test` by construction. The
disjointness check is applied to `train`, `calibration`, `selection` and `test`,
and `evaluation` is excluded — otherwise the subset relation would trip it.
The report always states which of the two sets a metric was computed on: clean
accuracy on `test`, ASR and defence deltas on `evaluation`.

## 4. Invariants verified by `scripts/audit_split.py`

Run against the study directory:

```
python scripts/audit_split.py --study runs/dase7508-low-rate-backdoor
```

The script fails loudly (`SystemExit`) if any of the following does not hold:

- `manifest.json`, all five `*.jsonl` files and `removed_duplicates.jsonl` exist;
- the `sha256` of every split matches `manifest["hashes"]`;
- per-split row counts and per-label counts match `manifest["counts"]`;
- `len(train) == train_size`, `len(calibration) == calibration_size`;
- `calibration + selection` equals `validation_size`;
- `evaluation ⊆ test`;
- `train`, `calibration`, `selection`, `test` are pairwise disjoint after
  normalisation (no sentence appears twice, even with different casing or
  Unicode width);
- the number of rows in `removed_duplicates.jsonl` equals
  `manifest["removed_duplicates"]`.

Current status on the study used for the report: **all invariants hold**.

## 5. Artefacts written by preparation

```
data/
  train.jsonl                 # 12000 rows, {id, text, label}
  calibration.jsonl           # 400
  selection.jsonl             # 1600
  test.jsonl                  # 7598
  evaluation.jsonl            # 1600
  removed_duplicates.jsonl    # 153
  training/<condition>.jsonl  # 30 files, 12000 rows each (clean or poisoned)
  audit_truth/<condition>.jsonl  # attack ground truth, empty for clean
  inventory.json              # condition table + trigger statistics + attack config
  manifest.json               # provenance, hashes, counts
```

Row ids are stable (`train-<n>`, `test-<n>`), which is what allows the attack
ground truth in `audit_truth/` to reference a training row without copying it.

## 6. Intended use and limits

- **Intended use**: measuring how far a 0.1 %–2 % poisoning budget pushes a
  small transformer towards a target label, and how much of that effect an
  inference-time defence (ONION) removes.
- **Not a benchmark of AG News itself.** The 12 000-row training split is a
  stratified subsample, so absolute accuracy is lower than full-data numbers and
  must not be compared with published AG News results.
- **Class balance is enforced, not natural.** Real news streams are not uniform
  over four topics; the balanced cut is a deliberate choice so that a per-class
  ASR is interpretable.
- **English only, headline-plus-body snippets.** Nothing here transfers to
  long-form or non-English text without re-checking the trigger realisation.
- **No personal data.** The corpus contains news text only; ids are synthetic
  row indices, not upstream identifiers.
