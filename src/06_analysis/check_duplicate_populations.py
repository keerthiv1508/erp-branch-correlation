#!/usr/bin/env python3
import json, os
from collections import defaultdict

DATA_PATH = os.path.expanduser("~/training_data.json")

with open(DATA_PATH) as f:
    data = json.load(f)

def analyze(records, label):
    vector_groups = defaultdict(list)
    for e in records:
        key = tuple(round(v, 6) for v in e["x"])
        vector_groups[key].append(e["y_label"])

    total = len(records)
    any_dup_examples = 0
    conflicted_examples = 0
    for key, labels in vector_groups.items():
        if len(labels) > 1:
            any_dup_examples += len(labels)
        n1 = sum(labels)
        n0 = len(labels) - n1
        if n0 > 0 and n1 > 0:
            conflicted_examples += len(labels)

    print(f"=== {label} ===")
    print(f"Total examples: {total}")
    print(f"Distinct vectors: {len(vector_groups)}")
    print(f"ANY duplicate (>=2 members, any label): {any_dup_examples} ({any_dup_examples/total:.3f})")
    print(f"OPPOSITELY-LABELLED duplicate: {conflicted_examples} ({conflicted_examples/total:.3f})")
    print()

non_augmented = [e for e in data if not e.get("augmented")]

analyze(non_augmented, "NON-AUGMENTED, 6,462 pairs  (current 42% check's population)")
analyze(data,          "FULL AUGMENTED, 12,924 examples  (current 87.1% check's population)")
