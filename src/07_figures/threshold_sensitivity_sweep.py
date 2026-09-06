#!/usr/bin/env python3
"""
Relabels examples at each candidate threshold (0.5, 0.6, 0.65, 0.7)
using |true correlation|, then evaluates LR and MLP under each.
Training code for both copied verbatim from check_full_metric_suite.py
and kfold_mlp_by_program.py -- only the relabelling and the outer
threshold loop are new.
"""
import json, os, random
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = os.path.expanduser("~/training_data.json")
SEED = 42
THRESHOLDS = [0.5, 0.6, 0.65, 0.7]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


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


def make_program_folds(programs, k, seed):
    rng = random.Random(seed)
    shuffled = list(programs)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for i, p in enumerate(shuffled):
        folds[i % k].append(p)
    return folds


def pooled_accuracy_lr(examples, folds):
    all_true, all_pred = [], []
    for i, test_programs in enumerate(folds):
        test_set = set(test_programs)
        train_ex = [e for e in examples if e["sample_index"] not in test_set]
        test_ex = [e for e in examples if e["sample_index"] in test_set]
        if not train_ex or not test_ex:
            continue
        X_train = np.array([e["x"] for e in train_ex])
        y_train = np.array([e["y_label"] for e in train_ex], dtype=float)
        X_test = np.array([e["x"] for e in test_ex])
        y_test = np.array([e["y_label"] for e in test_ex], dtype=float)
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        std[std == 0] = 1.0
        w, b = train_logistic_regression((X_train - mean) / std, y_train, lr=0.1, epochs=1000)
        probs = sigmoid(((X_test - mean) / std) @ w + b)
        all_true.extend(y_test.tolist())
        all_pred.extend((probs >= 0.5).astype(int).tolist())
    return (np.array(all_pred) == np.array(all_true)).mean()


def pooled_accuracy_mlp(examples, folds):
    all_true, all_pred = [], []
    for i, test_programs in enumerate(folds):
        test_set = set(test_programs)
        train_ex = [e for e in examples if e["sample_index"] not in test_set]
        test_ex = [e for e in examples if e["sample_index"] in test_set]
        if not train_ex or not test_ex:
            continue
        X_train = np.array([e["x"] for e in train_ex])
        y_train = np.array([e["y_label"] for e in train_ex], dtype=float)
        X_test = np.array([e["x"] for e in test_ex])
        y_test = np.array([e["y_label"] for e in test_ex], dtype=float)
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        std[std == 0] = 1.0
        model = SmallMLP(input_dim=X_train.shape[1], hidden_dim=32, seed=i)
        model.train((X_train - mean) / std, y_train, lr=0.05, epochs=2000)
        probs = model.predict_proba((X_test - mean) / std)
        all_true.extend(y_test.tolist())
        all_pred.extend((probs >= 0.5).astype(int).tolist())
    return (np.array(all_pred) == np.array(all_true)).mean()


def main():
    with open(DATA_PATH) as f:
        examples = json.load(f)

    programs = sorted(set(e["sample_index"] for e in examples))
    folds = make_program_folds(programs, 5, SEED)

    results = {"LR": {}, "MLP": {}}
    for t in THRESHOLDS:
        relabeled = []
        for e in examples:
            e2 = dict(e)
            e2["y_label"] = 1 if abs(e["y_correlation"]) >= t else 0
            relabeled.append(e2)

        lr_acc = pooled_accuracy_lr(relabeled, folds)
        mlp_acc = pooled_accuracy_mlp(relabeled, folds)
        results["LR"][t] = lr_acc
        results["MLP"][t] = mlp_acc
        print(f"threshold={t}: LR acc={lr_acc:.4f}  MLP acc={mlp_acc:.4f}", flush=True)

    with open("threshold_sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    for model, res in results.items():
        ts = sorted(res.keys())
        accs = [res[t] for t in ts]
        ax.plot(ts, accs, marker="o", linewidth=2, label=model)
        for t, a in zip(ts, accs):
            ax.annotate(f"{a*100:.1f}%", (t, a), textcoords="offset points", xytext=(0, 8), ha="center")

    ax.set_xlabel("Correlation threshold")
    ax.set_ylabel("Pooled accuracy")
    ax.set_title("Accuracy by correlation threshold")
    ax.set_xticks(THRESHOLDS)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figure6_threshold_sensitivity.png", dpi=200)
    print("\nSaved figure6_threshold_sensitivity.png")


if __name__ == "__main__":
    main()

# --- baseline check, run separately ---
