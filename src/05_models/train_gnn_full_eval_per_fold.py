#!/usr/bin/env python3
"""
Same as train_gnn_full_eval.py, verbatim, with one addition: per-fold
accuracy is computed and saved, needed for significance_test.py, which
currently holds GNN values from an older, superseded script. Training
logic, model, seeding: all unchanged from train_gnn_full_eval.py.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, log_loss, brier_score_loss, cohen_kappa_score
)

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

GRAPH_DATA_PATH = os.path.expanduser("~/graph_dataset.json")
SEED = 42
K = 5
HIDDEN_DIM = 32
EPOCHS = 40
LR = 0.01
BATCH_SIZE = 32


class BranchGNN(nn.Module):
    def __init__(self, in_dim=75, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, pair_list):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        logits = []
        for a, b in pair_list:
            combined = torch.cat([x[a], x[b]])
            logits.append(self.classifier(combined))
        if not logits:
            return torch.empty(0)
        return torch.cat(logits).view(-1)


def to_pyg_inputs(graph_entry):
    x = torch.tensor(graph_entry["node_embeddings"], dtype=torch.float32)
    edges = graph_entry["edges"]
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src = [e[0] for e in edges] + [e[1] for e in edges]
        dst = [e[1] for e in edges] + [e[0] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return x, edge_index


def make_program_folds(programs, k, seed):
    rng = random.Random(seed)
    shuffled = list(programs)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for i, p in enumerate(shuffled):
        folds[i % k].append(p)
    return folds


def main():
    with open(GRAPH_DATA_PATH) as f:
        graph_dataset = json.load(f)

    programs = sorted(set(g["sample_index"] for g in graph_dataset))
    folds = make_program_folds(programs, K, SEED)
    loss_fn = nn.BCEWithLogitsLoss()

    all_out_of_fold_probs = []
    all_out_of_fold_labels = []
    all_out_of_fold_true_r = []
    per_fold_accuracy = []  # NEW

    for fold_i, test_programs in enumerate(folds):
        test_set = set(test_programs)
        train_graphs = [g for g in graph_dataset if g["sample_index"] not in test_set and g["pairs"]]
        test_graphs = [g for g in graph_dataset if g["sample_index"] in test_set]
        if not train_graphs or not test_graphs:
            continue

        torch.manual_seed(SEED)
        model = BranchGNN(in_dim=75, hidden_dim=HIDDEN_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        t0 = time.time()
        model.train()
        for epoch in range(EPOCHS):
            random.Random(SEED + epoch).shuffle(train_graphs)
            batch_losses = []
            for i, g in enumerate(train_graphs):
                x, edge_index = to_pyg_inputs(g)
                pair_list = [(p["node_a"], p["node_b"]) for p in g["pairs"]]
                labels = torch.tensor([p["y_label"] for p in g["pairs"]], dtype=torch.float32)
                logits = model(x, edge_index, pair_list)
                loss = loss_fn(logits, labels)
                batch_losses.append(loss)
                if len(batch_losses) == BATCH_SIZE or i == len(train_graphs) - 1:
                    optimizer.zero_grad()
                    torch.stack(batch_losses).mean().backward()
                    optimizer.step()
                    batch_losses = []

        model.eval()
        fold_probs, fold_labels = [], []  # NEW: fold-local, for this fold's own accuracy
        with torch.no_grad():
            for g in test_graphs:
                if not g["pairs"]:
                    continue
                x, edge_index = to_pyg_inputs(g)
                pair_list = [(p["node_a"], p["node_b"]) for p in g["pairs"]]
                logits = model(x, edge_index, pair_list)
                probs = torch.sigmoid(logits).tolist()
                labels = [p["y_label"] for p in g["pairs"]]
                true_r = [p["y_correlation"] for p in g["pairs"]]

                fold_probs.extend(probs)          # NEW
                fold_labels.extend(labels)         # NEW
                all_out_of_fold_probs.extend(probs)
                all_out_of_fold_labels.extend(labels)
                all_out_of_fold_true_r.extend(true_r)

        fold_preds = (np.array(fold_probs) >= 0.5).astype(int)          # NEW
        fold_acc = (fold_preds == np.array(fold_labels)).mean()          # NEW
        per_fold_accuracy.append(float(fold_acc))                       # NEW

        print(f"Fold {fold_i+1} done, n_test={len(fold_labels)}, "
              f"acc={fold_acc:.4f}, time={time.time()-t0:.1f}s", flush=True)

    y_prob = np.array(all_out_of_fold_probs)
    y_true = np.array(all_out_of_fold_labels)
    y_pred = (y_prob >= 0.5).astype(int)
    true_r_abs = np.abs(np.array(all_out_of_fold_true_r))

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n" + "=" * 60)
    print("GNN FULL EVALUATION (out-of-fold predictions, all 5 folds combined)")
    print("=" * 60)
    print(f"\nPer-fold accuracy: {[round(a, 4) for a in per_fold_accuracy]}")
    print(f"Mean of per-fold accuracy: {np.mean(per_fold_accuracy):.4f}")

    print(f"\n-- Confusion matrix (threshold=0.5) --")
    print(f"                 Predicted correlated   Predicted uncorrelated")
    print(f"Actual correlated:      TP={tp:5d}              FN={fn:5d}")
    print(f"Actual uncorrelated:    FP={fp:5d}              TN={tn:5d}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"Accuracy (pooled): {(y_pred == y_true).mean():.4f}")

    print(f"\n-- Threshold-independent --")
    print(f"ROC-AUC:              {roc_auc_score(y_true, y_prob):.4f}")
    print(f"PR-AUC (avg prec):    {average_precision_score(y_true, y_prob):.4f}")
    print(f"Log loss:             {log_loss(y_true, y_prob):.4f}")
    print(f"Brier score:          {brier_score_loss(y_true, y_prob):.4f}")

    print(f"\n-- Threshold-dependent --")
    print(f"Matthews Corr Coef:   {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"Balanced accuracy:    {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Cohen's Kappa:        {cohen_kappa_score(y_true, y_pred):.4f}")

    print(f"\n-- Confidence vs actual correlation strength --")
    if HAS_SCIPY:
        rho, pval = spearmanr(y_prob, true_r_abs)
        print(f"Spearman(predicted_prob, true |r|): rho={rho:.4f}, p={pval:.2e}")

    print(f"\nBaseline reference: majority-class accuracy would be "
          f"{max(y_true.mean(), 1-y_true.mean()):.4f}")

    with open('gnn_full_eval_results.json', 'w') as f:
        json.dump({
            "per_fold_accuracy": per_fold_accuracy,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "pooled_accuracy": float((y_pred == y_true).mean()),
        }, f, indent=2)


if __name__ == "__main__":
    main()
