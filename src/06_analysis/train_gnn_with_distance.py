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
PAIRS_PATH = os.path.expanduser("~/pairs_multirun.json")
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
        # +1 input dim for the log-distance scalar, appended alongside the two node embeddings
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, pair_list):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        logits = []
        for a, b, dist in pair_list:
            dist_t = torch.tensor([dist], dtype=torch.float32)
            combined = torch.cat([x[a], x[b], dist_t])
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
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)

    dist_lookup = {}
    for p in pairs:
        line_a = p["branch_a"].get("src_line")
        line_b = p["branch_b"].get("src_line")
        if line_a is None or line_b is None:
            continue
        key = (p["sample_index"], p["function"], round(p["pearson_correlation"], 6))
        dist_lookup[key] = np.log1p(abs(line_a - line_b))

    # Attach distance to each pair in the graph dataset, skip pairs we can't resolve
    n_resolved, n_missing = 0, 0
    for g in graph_dataset:
        for p in g["pairs"]:
            key = (g["sample_index"], g["function"], round(p["y_correlation"], 6))
            dist = dist_lookup.get(key)
            if dist is None:
                n_missing += 1
                p["log_distance"] = 0.0  # neutral fallback, rare case
            else:
                n_resolved += 1
                p["log_distance"] = float(dist)
    print(f"Distance resolved for {n_resolved} pairs, missing for {n_missing} pairs")

    programs = sorted(set(g["sample_index"] for g in graph_dataset))
    folds = make_program_folds(programs, K, SEED)
    loss_fn = nn.BCEWithLogitsLoss()

    fold_accs = []
    fold_baselines = []

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
                pair_list = [(p["node_a"], p["node_b"], p["log_distance"]) for p in g["pairs"]]
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
        all_preds, all_labels = [], []
        with torch.no_grad():
            for g in test_graphs:
                if not g["pairs"]:
                    continue
                x, edge_index = to_pyg_inputs(g)
                pair_list = [(p["node_a"], p["node_b"], p["log_distance"]) for p in g["pairs"]]
                labels = [p["y_label"] for p in g["pairs"]]
                logits = model(x, edge_index, pair_list)
                preds = (torch.sigmoid(logits) >= 0.5).long().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = (all_preds == all_labels).mean()
        train_labels_flat = [p["y_label"] for g in train_graphs for p in g["pairs"]]
        majority = 1 if np.mean(train_labels_flat) >= 0.5 else 0
        baseline_acc = (np.full_like(all_labels, majority) == all_labels).mean()

        fold_accs.append(acc)
        fold_baselines.append(baseline_acc)
        print(f"Fold {fold_i+1}: acc={acc:.3f}, baseline={baseline_acc:.3f}, time={time.time()-t0:.1f}s "
              f"{'(beats baseline)' if acc > baseline_acc else '(does NOT beat baseline)'}")

    print("\n" + "=" * 50)
    print("GNN + LINE-DISTANCE K-FOLD SUMMARY")
    print("=" * 50)
    print(f"Mean accuracy: {np.mean(fold_accs):.4f} (std: {np.std(fold_accs):.4f})")
    print(f"Previous GNN (no distance) mean: 0.5991")
    print(f"Folds beating baseline: {sum(1 for a,b in zip(fold_accs, fold_baselines) if a > b)}/{len(fold_accs)}")


if __name__ == "__main__":
    main()
