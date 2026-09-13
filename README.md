# Statically Predicting Pairwise Branch Correlation from IR2Vec Embeddings

MSc Data Science - Extended Research Project (DATA72000, University of Manchester).

Can the correlation between two conditional branches be predicted **statically**,
from IR2Vec embeddings of the two branch instructions, without running the
program? Ground truth is dynamic: ExeBench programs are compiled, traced under
Intel Pin, and branch pairs labelled from the recorded outcome sequences.

**Result.** 87.1% of training examples share their exact 150-dimensional input
vector with an oppositely-labelled example, capping any model over that vector at
69.4% accuracy. All four architectures land between 0.581 and 0.599, none
significantly better than another. The representation, not the architecture, is
the binding constraint.

---

## Reproducing the reported results

Everything below runs from the shipped data. No compilation, tracing or ExeBench
access is needed.

### Setup

Scripts read their inputs from `$HOME` and are run from `$HOME`.

```bash
git clone https://github.com/keerthiv1508/erp-branch-correlation.git
cd erp-branch-correlation
cp data/*.json data/*.json.gz data/*.npz ~/
cd ~ && gunzip -f training_data.json.gz graph_dataset.json.gz
```

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r erp-branch-correlation/requirements.txt
```

Python 3.9.25, CPU only. No GPU is used at any stage.

### Train and evaluate

Seed 42 throughout; 5-fold cross-validation split at **program** level on
`sample_index`, identical folds for all four models.

```bash
cd ~
R=~/erp-branch-correlation/src

python3 $R/05_models/lr_per_fold_verify.py             # -> 0.581
python3 $R/05_models/mlp_per_fold_verify.py            # -> 0.592
python3 $R/05_models/train_lstm_pooled.py              # -> 0.587 
python3 $R/05_models/train_gnn_full_eval_per_fold.py   # -> 0.599  (~3 min/fold)

python3 $R/05_models/significance_test_updated.py      # Wilcoxon, paired folds
python3 $R/06_analysis/check_label_conflicts.py        # -> 87.1%, ceiling 69.4%
python3 $R/06_analysis/check_relational_feature_ceiling.py  # -> ceiling 79.8%
```

LSTM -  The script saves the pooled out-of-fold accuracy (0.5857). The reported 0.587 is the mean of the five per-fold accuracies, consistent with the other three models; those values print to the console but are not saved.

```bash
python3 $R/05_models/train_lstm_pooled.py 2>&1 | grep "fold acc"
```

`distance_vs_conflict.py` needs intermediate artefacts that are too large to ship
and will not run from a clone. Its output is in `results/`.

### Expected values

| Model | 5-fold accuracy |
|---|---|
| Majority-class baseline | 0.539 |
| Logistic regression | 0.581 |
| MLP | 0.592 |
| LSTM | 0.587 |
| GNN | 0.599 |

Wilcoxon signed-rank on paired per-fold accuracies (n = 5): GNN vs LR p = 0.125,
vs MLP p = 0.625, vs LSTM p = 0.313. Nothing significant.

Raw output for all reported figures is in `results/`.

---

## Rebuilding the corpus from ExeBench

Only needed to regenerate the datasets from scratch. Takes several hours and
requires LLVM 22.1.0, Intel Pin 3.28 and the ExeBench Python package.

These scripts carry absolute paths from the machine the study ran on and must be
edited before use:

| File | Line | Constant |
|---|---|---|
| `02_trace/scale_pipeline.py` | 14, 15, 17 | `EXEBENCH_PKG_DIR`, `PIN_ROOT`, `MAPPING_SCRIPT` |
| `03_mapping/map_branches.py` | 76 | `llvm-objdump` path |
| `04_features/attach_embeddings.py` | 11, 12 | `VOCAB_PATH`, `IR2VEC_BIN` |
| `04_features/build_graph_dataset.py` | 67 | `opt` path |

```bash
cd erp-branch-correlation/pin
make obj-intel64/branch_trace.so TARGET=intel64 PIN_ROOT=/path/to/pin-3.28
```

```bash
python3 src/02_trace/scale_pipeline.py --start 100 --count 42669
python3 src/04_features/attach_embeddings.py
python3 src/04_features/build_pairs_multirun.py
python3 src/04_features/build_training_data.py
python3 src/04_features/build_graph_dataset.py
```

Expect roughly 13,556 compiled entries, 6,462 pairs across 2,333 programs,
12,924 examples after symmetric augmentation. The compiled-entry count is
environment sensitive — a different Clang version gives a slightly different
number.

Mapping is **positional** and `num_cases` counts contiguous destination runs
rather than case arms.

---

## How pairs are labelled

`pin/branch_trace.cpp` records, for every conditional branch site reached,
whether it was taken.

`src/04_features/build_pairs_multirun.py` forms within-program branch pairs and
computes the Pearson correlation between the two branches' taken ratios, over the
runs in which both executed. No label is assigned at this stage.

The binary label is applied independently but identically in
`build_training_data.py:48` and `build_graph_dataset.py:173`:

```python
label = 1 if abs(r) >= 0.5 else 0
```

The threshold applies to `abs(r)`, so a strongly anti-correlated pair is labelled
correlated: the task is predicting whether two branches are statistically
related, not whether they agree. `CORRELATION_THRESHOLD` is set at the top of
each file and is a free parameter.

---

## Data

Source corpus: [ExeBench](https://huggingface.co/datasets/jordiae/exebench),
split `train_real_simple_io` (43,085 entries). Public, not redistributed here;
streamed via the `datasets` library.

Derived datasets are in `data/`. Intermediate per-program artefacts run to many
gigabytes and are not included.

## Layout

```
pin/             Intel Pin instrumentation tool
src/02_trace/    Compilation and tracing
src/03_mapping/  Binary-to-IR mapping
src/04_features/ Embeddings, pairing, dataset assembly
src/05_models/   LR, MLP, LSTM, GNN; significance test
src/06_analysis/ Label conflict and ceiling analysis
data/            Derived datasets (two gzipped)
results/         Raw per-fold and holdout output
figures/         Generated figures
```

This repository contains the scripts that produce the reported results.
Superseded pipeline variants, holdout and diagnostic scripts, and figure
generation code are excluded; their output is retained in `results/` and
`figures/`.

## Contact

Keerthivasan Ramasamy — keerthiv25uk@gmail.com
