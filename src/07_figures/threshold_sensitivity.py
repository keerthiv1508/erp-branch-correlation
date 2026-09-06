#!/usr/bin/env python3
import json
import os

PAIRS_PATH = os.path.expanduser("~/pairs_multirun.json")


def main():
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)

    correlations = [p["pearson_correlation"] for p in pairs]
    n = len(correlations)
    print(f"Total pairs: {n}")
    print(f"Correlation value range: [{min(correlations):.3f}, {max(correlations):.3f}]")
    print()

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"{'threshold':>10} | {'n_positive':>10} | {'n_negative':>10} | {'%% positive':>12}")
    print("-" * 50)
    for t in thresholds:
        n_pos = sum(1 for c in correlations if abs(c) >= t)
        n_neg = n - n_pos
        pct = 100 * n_pos / n if n else 0
        print(f"{t:>10.2f} | {n_pos:>10} | {n_neg:>10} | {pct:>11.1f}%")

    print()
    print("Distribution of |correlation| values (histogram, 10 bins):")
    abs_corrs = sorted(abs(c) for c in correlations)
    bin_edges = [i / 10 for i in range(11)]
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        count = sum(1 for c in abs_corrs if lo <= c < hi or (i == 9 and c == hi))
        bar = "#" * count
        print(f"  [{lo:.1f}, {hi:.1f}): {count:>4} {bar}")

    print()
    print("If the %% positive column changes sharply between adjacent thresholds,")
    print("the 0.5 cutoff is sitting in a dense/unstable region of the data --")
    print("worth picking a threshold in a flatter part of the distribution instead,")
    print("or using the continuous value directly (regression) rather than a cutoff.")


if __name__ == "__main__":
    main()
