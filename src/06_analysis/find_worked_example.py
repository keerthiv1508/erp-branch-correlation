#!/usr/bin/env python3
import json
import os
from collections import defaultdict

DATA_PATH = os.path.expanduser("~/training_data.json")

with open(DATA_PATH) as f:
    data = json.load(f)

vector_groups = defaultdict(list)
for e in data:
    key = tuple(round(v, 6) for v in e["x"])
    vector_groups[key].append({
        "sample_index": e["sample_index"],
        "function": e["function"],
        "augmented": e["augmented"],
        "label": e["y_label"],
        "correlation": e["y_correlation"],
    })

conflicted_groups = [
    entries for entries in vector_groups.values()
    if len(set(e["label"] for e in entries)) > 1
]
biggest = max(conflicted_groups, key=len)

print(f"Largest conflicted group: {len(biggest)} examples")
print(f"Distinct programs (sample_index): {len(set(e['sample_index'] for e in biggest))}")
print(f"Distinct functions: {len(set(e['function'] for e in biggest))}")
print()
print("First 15 members:")
for e in biggest[:15]:
    print(f"  sample={e['sample_index']:>8}  function={e['function']:<30}  augmented={e['augmented']!s:<5}  label={e['label']}  r={e['correlation']:.3f}")
