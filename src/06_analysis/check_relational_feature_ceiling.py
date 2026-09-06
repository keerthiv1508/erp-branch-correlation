#!/usr/bin/env python3
import json
import os
from collections import defaultdict

DATA_PATH = os.path.expanduser("~/training_data.json")
PAIRS_PATH = os.path.expanduser("~/pairs_multirun.json")

with open(DATA_PATH) as f:
    data = json.load(f)
with open(PAIRS_PATH) as f:
    pairs = json.load(f)

# Index pairs for lookup: (sample_index, function, rounded correlation) -> line_a, line_b
pair_lookup = {}
for p in pairs:
    key = (p["sample_index"], p["function"], round(p["pearson_correlation"], 6))
    pair_lookup[key] = p

def theoretical_ceiling(get_key_fn, label="baseline"):
    vector_labels = defaultdict(list)
    skipped = 0
    for e in data:
        key = get_key_fn(e)
        if key is None:
            skipped += 1
            continue
        vector_labels[key].append(e["y_label"])

    total = sum(len(v) for v in vector_labels.values())
    majority_correct = sum(max(sum(v), len(v) - sum(v)) for v in vector_labels.values())
    n_conflicted = sum(1 for v in vector_labels.values() if 0 < sum(v) < len(v))

    print(f"--- {label} ---")
    print(f"Examples used: {total} (skipped: {skipped})")
    print(f"Distinct keys: {len(vector_labels)}")
    print(f"Conflicted groups: {n_conflicted}")
    print(f"Theoretical max accuracy: {majority_correct/total:.4f}")
    print()
    return majority_correct / total

# Baseline: original 150D vector only (should reproduce 0.694)
def key_original(e):
    return tuple(round(v, 6) for v in e["x"])

ceiling_original = theoretical_ceiling(key_original, "Original 150D (no relational feature)")

# Extended: 150D vector + line-distance between the two branches
def key_with_line_distance(e):
    key = (e["sample_index"], e["function"], round(e["y_correlation"], 6))
    p = pair_lookup.get(key)
    if p is None:
        return None
    line_a = p["branch_a"].get("src_line")
    line_b = p["branch_b"].get("src_line")
    if line_a is None or line_b is None:
        return None
    distance = abs(line_a - line_b)
    return tuple(round(v, 6) for v in e["x"]) + (distance,)

ceiling_with_distance = theoretical_ceiling(key_with_line_distance, "150D + line-distance feature")

print("=" * 60)
print(f"Ceiling WITHOUT relational feature: {ceiling_original:.4f}")
print(f"Ceiling WITH line-distance added:   {ceiling_with_distance:.4f}")
print(f"Improvement: {ceiling_with_distance - ceiling_original:+.4f}")
