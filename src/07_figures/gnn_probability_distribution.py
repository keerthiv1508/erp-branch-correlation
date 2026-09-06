#!/usr/bin/env python3
"""
Same training as train_gnn_full_eval_per_fold.py, verbatim. Adds one
thing: a histogram of predicted probability, split by whether the
prediction was correct, using the same pooled out-of-fold predictions.
"""
import json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

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

    all_probs, all_labels = [], []

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
        with torch.no_grad():
            for g in test_graphs:
                if not g["pairs"]:
                    continue
                x, edge_index = to_pyg_inputs(g)
                pair_list = [(p["node_a"], p["node_b"]) for p in g["pairs"]]
                logits = model(x, edge_index, pair_list)
                probs = torch.sigmoid(logits).tolist()
                all_probs.extend(probs)
                all_labels.extend([p["y_label"] for p in g["pairs"]])

        print(f"Fold {fold_i+1} done, time={time.time()-t0:.1f}s", flush=True)

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)
    correct = (y_pred == y_true)

    with open("gnn_probabilities.json", "w") as f:
        json.dump({"probs": all_probs, "labels": all_labels, "correct": correct.tolist()}, f)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = [i / 20 for i in range(21)]
    ax.hist(y_prob[correct], bins=bins, alpha=0.6, label="Correct predictions")
    ax.hist(y_prob[~correct], bins=bins, alpha=0.6, label="Incorrect predictions")
    ax.axvline(0.5, linestyle="--", color="grey", alpha=0.7)
    ax.set_xlabel("Predicted probability of CORRELATED")
    ax.set_ylabel("Number of examples")
    ax.set_title("Predicted probability distribution (GNN)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figure7_probability_distribution.png", dpi=200)
    print(f"\nAccuracy check: {correct.mean():.4f} (should match 0.5992 from earlier)")
    print("Saved figure7_probability_distribution.png")


if __name__ == "__main__":
    main()
