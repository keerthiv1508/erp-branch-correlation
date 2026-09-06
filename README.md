# Statically Predicting Pairwise Branch Correlation from IR2Vec Embeddings

MSc Data Science Extended Research Project (DATA72000, University of Manchester).

Can the correlation between two conditional branches be predicted **statically**,
from IR2Vec embeddings of the two branch instructions, without running the
program? Ground truth is dynamic: ExeBench programs are compiled, traced under
Intel Pin, and branch pairs labelled from the recorded outcome sequences.

**Result.** 87.1% of training examples share their exact 150-dimensional input
vector with an oppositely-labelled example, capping any model over that vector at
69.4% accuracy. All four architectures land between 0.581 and 0.599, none
significantly better than another. The representation, not the architecture, is
the binding constraint.

Hyperparameters, method detail and the full result set are in the technical
appendix submitted alongside this repository.

---

## Reproducing the reported results

**Everything in stages 4 and 5 below runs from the shipped data. No compilation,
tracing or ExeBench access is needed.**

### Setup

Scripts read their inputs from `$HOME` and are run from `$HOME`. Copy the data
files there once:

```bash
git clone https://github.com/keerthiv1508/erp-branch-correlation.git
cd erp-branch-correlation
cp data/*.json data/*.json.gz data/*.npz ~/
cd ~ && gunzip -f training_data.json.gz graph_dataset.json.gz
```

Leaves `training_data.json`, `graph_dataset.json`, `pairs_multirun.json`,
`holdout_examples.json` and `lstm_data_by_program.npz` in `$HOME`.

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r erp-branch-correlation/requirements.txt
```

Python 3.9.25, CPU only. No GPU is used at any stage.

### Stage 4 — train and evaluate

Run from `$HOME`. Seed 42 throughout; 5-fold cross-validation split at **program**
level on `sample_index`, identical folds for all four models.

```bash
cd ~
R=~/erp-branch-correlation/src

python3 $R/05_models/lr_per_fold_verify.py             # -> 0.5809
python3 $R/05_models/mlp_per_fold_verify.py            # -> 0.5916
python3 $R/05_models/train_lstm_pooled.py              # -> 0.587
python3 $R/05_models/train_gnn_full_eval_per_fold.py   # -> 0.599  (~3 min/fold)

python3 $R/05_models/lr_holdout_verify.py              # -> 0.694
python3 $R/05_models/mlp_holdout_and_lr_strict.py      # -> 0.673
python3 $R/05_models/gnn_holdout_and_strict.py         # -> 0.694, recall 0.917
python3 $R/05_models/significance_test_updated.py      # Wilcoxon, paired folds
```

`prep_lstm_data_by_program.py` regenerates `lstm_data_by_program.npz`; the file is
shipped, so this is optional.

### Stage 5 — analysis and figures

```bash
python3 $R/06_analysis/check_label_conflicts.py           # -> 87.1%, ceiling 69.4%
python3 $R/06_analysis/check_relational_feature_ceiling.py # -> ceiling 79.8%
python3 $R/06_analysis/check_duplicate_exact.py
python3 $R/06_analysis/check_emb_collapse.py
python3 $R/06_analysis/check_programs_functions.py        # -> 2,333 / 1,767
python3 $R/06_analysis/check_confusion_matrix.py
python3 $R/06_analysis/check_confusion_matrix_strict.py
python3 $R/06_analysis/show_ceiling.py
python3 $R/06_analysis/find_worked_example.py

python3 $R/07_figures/threshold_sensitivity_sweep.py
python3 $R/07_figures/threshold_sensitivity.py
python3 $R/07_figures/gnn_probability_distribution.py
python3 $R/07_figures/train_gnn_loss_curve.py
```

Three scripts in `06_analysis/` need intermediate artefacts that are **not**
shipped (too large) and will not run from a clone: `distance_vs_conflict.py`,
`check_collapse_mechanism_split.py`, `check_failure_buckets.py` and
`check_failure_categories.py`. Their output is included in `results/`. Rebuilding
their inputs requires stages 1–3.

### Expected values

| Model | 5-fold accuracy | Holdout accuracy |
|---|---|---|
| Majority-class baseline | 0.539 | — |
| Logistic regression | 0.581 | 0.694 |
| MLP | 0.592 | 0.673 |
| LSTM | 0.587 | — |
| GNN | 0.599 | 0.694 |

GNN on the strict-correlation class: recall 0.917, precision 0.324 (12 of 49
holdout pairs strictly correlated).

Wilcoxon signed-rank on paired per-fold accuracies (n = 5): GNN vs LR p = 0.125,
vs MLP p = 0.625, vs LSTM p = 0.313. Nothing significant.

Representation ceiling: 87.1% of examples conflicted, ceiling 69.4%. Adding a
relational distance feature raises the ceiling to 79.8% with no accuracy gain.

Raw output for all of the above is in `results/`.

---

## Rebuilding the corpus from ExeBench (stages 1–3)

Only needed to regenerate the datasets from scratch. Takes several hours and
requires LLVM 22.1.0, Intel Pin 3.28, and the ExeBench Python package.

**These scripts carry absolute paths from the machine the study ran on and must
be edited before use:**

| File | Line | Constant |
|---|---|---|
| `02_trace/scale_pipeline.py` | 14, 15, 17 | `EXEBENCH_PKG_DIR`, `PIN_ROOT`, `MAPPING_SCRIPT` |
| `02_trace/clang_batch.py` | 6, 8, 10 | `MAPPING_DIR`, `PIN_ROOT`, `EXEBENCH_PKG_DIR` |
| `02_trace/clang_compile_one.py` | 8 | `_EXEBENCH_PKG_DIR` |
| `02_trace/multi_run_trace.py` | 11 | `PIN_ROOT` |
| `02_trace/run_pin_batch_parallel.py` | 17, 19 | `PIN_ROOT`, `OUTPUT_DIR` |
| `03_mapping/map_branches.py` | 76 | `llvm-objdump` path |
| `03_mapping/export_mapping_json_batch.py` | 20 | `OUT_DIR_DEFAULT` |
| `03_mapping/run_mapping_batch.py` | 19 | `SAMPLE_DIR` |
| `03_mapping/verify_mismatches.py` | 11 | `TRACE_DIR` |
| `04_features/attach_embeddings.py` | 11, 12 | `VOCAB_PATH`, `IR2VEC_BIN` |
| `04_features/build_graph_dataset.py` | 67 | `opt` path |
| `06_analysis/distance_vs_conflict.py` | 150 | output path |
| `06_analysis/check_failure_*.py` | 3 | ExeBench `sys.path` |

Build the Pin tool first:

```bash
cd erp-branch-correlation/pin
make obj-intel64/branch_trace.so TARGET=intel64 PIN_ROOT=/path/to/pin-3.28
```

Then:

```bash
python3 src/02_trace/scale_pipeline.py --start 100 --count 42669
python3 src/03_mapping/export_mapping_json_batch.py
python3 src/04_features/attach_embeddings.py
python3 src/04_features/build_pairs_multirun.py
python3 src/04_features/build_training_data.py
python3 src/04_features/build_graph_dataset.py
```

Expect roughly 13,556 compiled entries, 36,276 mapped branches, 6,462 pairs
across 2,333 programs and 1,767 functions, 12,924 examples after symmetric
augmentation. The compiled-entry count is environment sensitive — a different
Clang version gives a slightly different number, with small shifts downstream.

Mapping is **positional** and `num_cases` counts contiguous destination runs
rather than case arms. Read appendix §4 before reading this code.

---

## Data

Source corpus: [ExeBench](https://huggingface.co/datasets/jordiae/exebench),
split `train_real_simple_io` (43,085 entries). Public, not redistributed here;
streamed via the `datasets` library. This split is required because entries in
other splits carry no runnable I/O.

Derived datasets are included in `data/`. Intermediate per-program artefacts
(`clang_pipeline_test/`, `mapping_json_clang/`, `mapping_json_embedded/`) are not
— they run to many gigabytes.

## Layout

```
pin/             Intel Pin instrumentation tool (branch_trace.cpp)
src/02_trace/    Acquisition, compilation, tracing
src/03_mapping/  Binary-to-IR mapping
src/04_features/ Embeddings, pairing, dataset assembly
src/05_models/   LR, MLP, LSTM, GNN; holdout; significance test
src/06_analysis/ Label conflict, ceiling, confusion matrices, bucketing
src/07_figures/  Figure generation
data/            Derived datasets (two gzipped)
results/         Raw per-fold and holdout output
figures/         Generated figures
```

Superseded code is excluded: earlier GNN training variants, the earlier
significance script, all `_v2`/`_v3` dataset variants, and two exploratory
acquisition scripts that do not build the corpus. Nothing removed contributes a
reported number.

## Citation

Armengol-Estapé et al., *ExeBench: an ML-scale dataset of executable C
functions*, MAPS 2022.
VenkataKeerthy et al., *IR2Vec: LLVM IR Based Scalable Program Embeddings*,
TACO 2020.
