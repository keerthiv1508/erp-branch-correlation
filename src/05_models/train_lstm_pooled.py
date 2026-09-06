#!/usr/bin/env python3
"""
LSTM retrained with the same program-level fold split used by LR, MLP,
and GNN (make_program_folds, seeded shuffle), replacing GroupKFold so
all four models are tested on identical held-out programs per fold.
Also fixes the missing seed and pools predictions for a confusion matrix.
Model architecture and train_fold logic copied verbatim from
train_lstm_kfold_by_program.py.
"""
import json, os, random, time
import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)


class PairLSTM(nn.Module):
    def __init__(self, input_dim=75, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h_n[-1])).squeeze(-1)


def train_fold(X_train, y_train, epochs=1000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = PairLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train)

    for e in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        if e % 200 == 0:
            print(f"  epoch {e}, loss {loss.item():.4f}", flush=True)
    return model


def make_program_folds(programs, k, seed):
    rng = random.Random(seed)
    shuffled = list(programs)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for i, p in enumerate(shuffled):
        folds[i % k].append(p)
    return folds


def main():
    d = np.load('lstm_data_by_program.npz', allow_pickle=True)
    X, y, groups = d['X'], d['y'], d['groups']

    programs = sorted(set(groups.tolist()))
    print(f"Total programs: {len(programs)}, total examples: {len(X)}")
    folds = make_program_folds(programs, 5, SEED)

    all_true, all_prob = [], []

    for fold_i, test_programs in enumerate(folds):
        test_set = set(test_programs)
        train_idx = [i for i, g in enumerate(groups) if g not in test_set]
        test_idx = [i for i, g in enumerate(groups) if g in test_set]
        if not train_idx or not test_idx:
            continue

        print(f"\nFold {fold_i+1}/5: n_train={len(train_idx)}, n_test={len(test_idx)}", flush=True)
        t0 = time.time()

        model = train_fold(X[train_idx].astype(np.float32), y[train_idx].astype(np.float32),
                            epochs=1000, lr=1e-3, seed=fold_i)

        with torch.no_grad():
            probs = model(torch.tensor(X[test_idx].astype(np.float32))).numpy()

        all_true.extend(y[test_idx].tolist())
        all_prob.extend(probs.tolist())

        acc = ((probs >= 0.5).astype(int) == y[test_idx]).mean()
        print(f"  fold acc={acc:.4f} ({time.time()-t0:.1f}s)", flush=True)

    y_true = np.array(all_true)
    y_pred = (np.array(all_prob) >= 0.5).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    print("\n" + "=" * 60)
    print("LSTM, PROGRAM-LEVEL SPLIT (pooled out-of-fold predictions)")
    print("=" * 60)
    print(f"                 Predicted correlated   Predicted uncorrelated")
    print(f"Actual correlated:      TP={tp:5d}              FN={fn:5d}")
    print(f"Actual uncorrelated:    FP={fp:5d}              TN={tn:5d}")
    print(f"Accuracy (pooled): {(y_pred == y_true).mean():.4f}")

    with open('lstm_program_split_results.json', 'w') as f:
        json.dump({"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "accuracy": float((y_pred == y_true).mean())}, f, indent=2)

if __name__ == "__main__":
    main()
