#!/usr/bin/env python3
import json, os
from collections import defaultdict

DATA_PATH = os.path.expanduser("~/training_data.json")

with open(DATA_PATH) as f:
    data = json.load(f)

non_augmented = [e for e in data if not e.get("augmented")]

for precision, label in [(None, "EXACT (no rounding)"), (9, "round to 9dp"), (6, "round to 6dp (current method)")]:
    vector_groups = defaultdict(list)
    for e in non_augmented:
        if precision is None:
            key = tuple(e["x"])
        else:
            key = tuple(round(v, precision) for v in e["x"])
        vector_groups[key].append(e["y_label"])

    total = len(non_augmented)
    any_dup = sum(len(v) for v in vector_groups.values() if len(v) > 1)
    print(f"{label}: distinct vectors={len(vector_groups)}, ANY duplicate={any_dup} ({any_dup/total:.3f})")
