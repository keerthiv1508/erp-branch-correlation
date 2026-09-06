#!/usr/bin/env python3
"""
Part 1: MLP holdout, same sample_index split as LR and GNN (SEED=42,
N_HOLDOUT_PROGRAMS=20, confirmed identical program set). MLP training
copied verbatim from mlp_pooled_confusion.py / kfold_mlp_by_program.py.

Part 2: LR rerun with the strict recall addition -- recall computed only
on examples with |true_correlation| >= 0.9, matching GNN's definition
in gnn_holdout_and_strict.py, so recall is genuinely comparable across
all three models, not just accuracy.
"""
import json, os, random
import numpy as np

DATA_PATH = os.path.expanduser("~/training_data.json")
SEED = 42
N_HOLDOUT_PROGRAMS = 20
STRICT_THRESHOLD = 0.9


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


class SmallMLP:
    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim,))
        self.b2 = 0.0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return sigmoid(z2), (X, z1, a1, z2)

    def train(self, X, y, lr=0.05, epochs=2000, l2=0.001):
        n = X.shape[0]
        for _ in range(epochs):
            preds, (X_c, z1, a1, z2) = self.forward(X)
            d_z2 = (preds - y) / n
            d_W2 = a1.T @ d_z2 + l2 * self.W2
            d_b2 = d_z2.sum()
            d_a1 = np.outer(d_z2, self.W2)
            d_z1 = d_a1 * relu_deriv(z1)
            d_W1 = X_c.T @ d_z1 + l2 * self.W1
            d_b1 = d_z1.sum(axis=0)
            self.W2 -= lr * d_W2; self.b2 -= lr * d_b2
            self.W1 -= lr * d_W1; self.b1 -= lr * d_b1

    def predict_proba(self, X):
        preds, _ = self.forward(X)
        return preds


def train_logistic_regression(X, y, lr=0.1, epochs=1000, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        preds = sigmoid(z)
        error = preds - y
        w -= lr * (X.T @ error / n)
        b -= lr * error.mean()
    return w, b


def strict_recall(y_hold_pred, true_r_abs, threshold):
    strict_true = (true_r_abs >= threshold).astype(int)
    tp = int(((y_hold_pred == 1) & (strict_true == 1)).sum())
    fn = int(((y_hold_pred == 0) & (strict_true == 1)).sum())
    n_strict = int(strict_true.sum())
    recall = tp / (tp + fn) if (tp + fn) else 0
    return recall, n_strict, tp, fn


def main():
    with open(DATA_PATH) as f:
        examples = json.load(f)

    all_programs = sorted(set(e["sample_index"] for e in examples))
    rng = random.Random(SEED)
    shuffled = list(all_programs)
    rng.shuffle(shuffled)
    holdout_programs = set(shuffled[:N_HOLDOUT_PROGRAMS])

    train_examples = [e for e in examples if e["sample_index"] not in holdout_programs]
    holdout_examples = [e for e in examples if e["sample_index"] in holdout_programs]

    X_train = np.array([e["x"] for e in train_examples])
    y_train = np.array([e["y_label"] for e in train_examples], dtype=float)
    X_hold = np.array([e["x"] for e in holdout_examples])
    y_hold = np.array([e["y_label"] for e in holdout_examples], dtype=float)
    true_r_abs = np.abs(np.array([e["y_correlation"] for e in holdout_examples]))

    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_norm = (X_train - mean) / std
    X_hold_norm = (X_hold - mean) / std

    print("=" * 60)
    print("LR (rerun, with strict recall added)")
    print("=" * 60)
    w, b = train_logistic_regression(X_train_norm, y_train, lr=0.1, epochs=1000)
    lr_probs = sigmoid(X_hold_norm @ w + b)
    lr_pred = (lr_probs >= 0.5).astype(int)
    lr_acc = (lr_pred == y_hold).mean()
    lr_strict_recall, n_strict, tp, fn = strict_recall(lr_pred, true_r_abs, STRICT_THRESHOLD)
    print(f"Accuracy: {lr_acc:.4f}")
    print(f"Strict-positive examples (|r|>={STRICT_THRESHOLD}): {n_strict}")
    print(f"Strict recall: {lr_strict_recall:.4f} (TP={tp}, FN={fn})")

    print("\n" + "=" * 60)
    print("MLP (new, sample_index split, identical 20 programs)")
    print("=" * 60)
    model = SmallMLP(input_dim=X_train.shape[1], hidden_dim=32, seed=0)
    model.train(X_train_norm, y_train, lr=0.05, epochs=2000)
    mlp_probs = model.predict_proba(X_hold_norm)
    mlp_pred = (mlp_probs >= 0.5).astype(int)
    mlp_acc = (mlp_pred == y_hold).mean()
    mlp_strict_recall, n_strict2, tp2, fn2 = strict_recall(mlp_pred, true_r_abs, STRICT_THRESHOLD)
    print(f"Accuracy: {mlp_acc:.4f}")
    print(f"Strict-positive examples (|r|>={STRICT_THRESHOLD}): {n_strict2}")
    print(f"Strict recall: {mlp_strict_recall:.4f} (TP={tp2}, FN={fn2})")

    print("\n" + "=" * 60)
    print("SUMMARY -- all three on the identical 20 held-out programs")
    print("=" * 60)
    print(f"LR : accuracy={lr_acc:.4f}  strict_recall={lr_strict_recall:.4f}")
    print(f"MLP: accuracy={mlp_acc:.4f}  strict_recall={mlp_strict_recall:.4f}")
    print(f"GNN: accuracy=0.6940 (already reported)  strict_recall=0.9170 (already reported)")

    with open("holdout_all_models.json", "w") as f:
        json.dump({
            "lr": {"accuracy": float(lr_acc), "strict_recall": float(lr_strict_recall)},
            "mlp": {"accuracy": float(mlp_acc), "strict_recall": float(mlp_strict_recall)},
        }, f, indent=2)


if __name__ == "__main__":
    main()
