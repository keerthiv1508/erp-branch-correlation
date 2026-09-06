#!/usr/bin/env python3
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

GRAPH_DATA_PATH = os.path.expanduser("~/graph_dataset.json")
SEED = 42
N_HOLDOUT_PROGRAMS = 20
HIDDEN_DIM = 32
EPOCHS = 40
LR = 0.01
BATCH_SIZE = 32
STRICT_THRESHOLD = 0.9


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


def main():
    with open(GRAPH_DATA_PATH) as f:
        graph_dataset = json.load(f)

    programs = sorted(set(g["sample_index"] for g in graph_dataset))
    rng = random.Random(SEED)
    shuffled = list(programs)
    rng.shuffle(shuffled)
    holdout_programs = set(shuffled[:N_HOLDOUT_PROGRAMS])

    train_graphs = [g for g in graph_dataset if g["sample_index"] not in holdout_programs and g["pairs"]]
    holdout_graphs = [g for g in graph_dataset if g["sample_index"] in holdout_programs]

    n_holdout_pairs = sum(len(g["pairs"]) for g in holdout_graphs)
    print(f"Training on {len(train_graphs)} function-graphs")
    print(f"Genuine holdout: {len(holdout_graphs)} function-graphs, {n_holdout_pairs} branch pairs, "
          f"{len(holdout_programs)} programs never touched during training")

    torch.manual_seed(SEED)
    model = BranchGNN(in_dim=75, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

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
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch} done")

    print(f"Training time: {time.time()-t0:.1f}s")

    model.eval()
    all_probs, all_labels, all_true_r = [], [], []
    with torch.no_grad():
        for g in holdout_graphs:
            if not g["pairs"]:
                continue
            x, edge_index = to_pyg_inputs(g)
            pair_list = [(p["node_a"], p["node_b"]) for p in g["pairs"]]
            logits = model(x, edge_index, pair_list)
            probs = torch.sigmoid(logits).tolist()
            all_probs.extend(probs)
            all_labels.extend([p["y_label"] for p in g["pairs"]])
            all_true_r.extend([p["y_correlation"] for p in g["pairs"]])

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)
    true_r_abs = np.abs(np.array(all_true_r))

    acc = (y_pred == y_true).mean()
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n" + "=" * 50)
    print(f"GENUINE HOLDOUT RESULT ({len(holdout_programs)} unseen programs)")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")

    # Strict threshold confusion matrix
    strict_true = (true_r_abs >= STRICT_THRESHOLD).astype(int)
    s_tp = int(((y_pred == 1) & (strict_true == 1)).sum())
    s_tn = int(((y_pred == 0) & (strict_true == 0)).sum())
    s_fp = int(((y_pred == 1) & (strict_true == 0)).sum())
    s_fn = int(((y_pred == 0) & (strict_true == 1)).sum())
    s_precision = s_tp / (s_tp + s_fp) if (s_tp + s_fp) else 0
    s_recall = s_tp / (s_tp + s_fn) if (s_tp + s_fn) else 0

    print(f"\n=== STRICT THRESHOLD (|r| >= {STRICT_THRESHOLD}) on holdout ===")
    print(f"Strict-positive examples in holdout: {int(strict_true.sum())}/{len(strict_true)}")
    print(f"TP={s_tp} TN={s_tn} FP={s_fp} FN={s_fn}")
    print(f"Precision: {s_precision:.4f}  Recall: {s_recall:.4f}")


if __name__ == "__main__":
    main()
