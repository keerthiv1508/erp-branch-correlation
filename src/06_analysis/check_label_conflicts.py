#!/usr/bin/env python3
import json
import os
from collections import defaultdict

DATA_PATH = os.path.expanduser("~/training_data.json")

with open(DATA_PATH) as f:
    data = json.load(f)

vector_labels = defaultdict(list)
for e in data:
    key = tuple(round(v, 6) for v in e["x"])
    vector_labels[key].append(e["y_label"])

n_pure = 0          # all instances of this vector agree on label
n_conflicted = 0     # this vector has both label 0 and label 1 among its instances
examples_in_pure = 0
examples_in_conflicted = 0
majority_correct = 0

for key, labels in vector_labels.items():
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n0 > 0 and n1 > 0:
        n_conflicted += 1
        examples_in_conflicted += len(labels)
    else:
        n_pure += 1
        examples_in_pure += len(labels)
    majority_correct += max(n0, n1)

total = len(data)
print(f"Total examples: {total}")
print(f"Distinct vectors: {len(vector_labels)}")
print()
print(f"Vectors with a SINGLE consistent label (pure): {n_pure}  ({examples_in_pure} examples)")
print(f"Vectors with CONFLICTING labels (same input, different labels): {n_conflicted}  ({examples_in_conflicted} examples)")
print()
print(f"Fraction of examples living in a conflicted vector: {examples_in_conflicted/total:.3f}")
print()
print(f"THEORETICAL MAX ACCURACY (best any model could ever do on this exact")
print(f"feature representation, by majority-voting each duplicate group): {majority_correct/total:.3f}")

# Show a few worst conflicted examples (biggest groups, most balanced conflict)
print()
print("Worst conflicted vectors (large groups, close to 50/50 split):")
conflict_list = []
for key, labels in vector_labels.items():
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n0 > 0 and n1 > 0:
        balance = min(n0, n1) / len(labels)
        conflict_list.append((len(labels), balance, n0, n1))
conflict_list.sort(key=lambda t: (-t[0] * t[1]))
for size, balance, n0, n1 in conflict_list[:10]:
    print(f"  group size={size}, label=0 count={n0}, label=1 count={n1}, balance={balance:.2f}")
