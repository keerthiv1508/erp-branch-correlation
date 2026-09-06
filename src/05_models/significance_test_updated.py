#!/usr/bin/env python3
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# Per-fold accuracies, same program-grouped folds (seed=42) across all four models
lr   = np.array([0.515, 0.586, 0.624, 0.544, 0.636])
mlp  = np.array([0.543, 0.577, 0.649, 0.561, 0.627])
lstm = np.array([0.5437, 0.5684, 0.6274, 0.5584, 0.6378])  # UPDATED: program-level split
gnn  = np.array([0.5485, 0.6132, 0.6354, 0.5720, 0.6265])  # confirmed matching current script

models = {"LR": lr, "MLP": mlp, "LSTM": lstm}

print("Paired comparison: GNN vs each other model (same 5 folds)\n")
for name, arr in models.items():
    diff = gnn - arr
    t_stat, t_p = ttest_rel(gnn, arr)
    try:
        w_stat, w_p = wilcoxon(gnn, arr)
    except ValueError:
        w_stat, w_p = None, None
    print(f"GNN vs {name}:")
    print(f"  Per-fold differences (GNN - {name}): {diff.round(4).tolist()}")
    print(f"  Mean difference: {diff.mean():.4f}")
    print(f"  Paired t-test:      t={t_stat:.3f}, p={t_p:.4f}")
    if w_p is not None:
        print(f"  Wilcoxon signed-rank: p={w_p:.4f}")
    print(f"  {'SIGNIFICANT at p<0.05' if t_p < 0.05 else 'NOT significant at p<0.05'}")
    print()

print("Note: with only n=5 paired folds, statistical power is inherently very low --")
print("a non-significant result here does not mean there is no real difference,")
print("only that 5 folds cannot reliably detect one this size.")
