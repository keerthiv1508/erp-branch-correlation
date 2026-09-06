#!/usr/bin/env python3
import json, os
from collections import defaultdict
import numpy as np

DATA_PATH = os.path.expanduser("~/training_data.json")

with open(DATA_PATH) as f:
    data = json.load(f)

nonaug = [e for e in data if not e.get("augmented")]
print(f"Total non-augmented examples: {len(nonaug)}")

collapsed = []
not_collapsed = []
for e in nonaug:
    x = np.array(e["x"])
    emb_a, emb_b = x[:75], x[75:]
    if np.allclose(emb_a, emb_b):
        collapsed.append(e)
    else:
        not_collapsed.append(e)

print(f"emb_a == emb_b (same example, two branches identical): {len(collapsed)} "
      f"({len(collapsed)/len(nonaug):.1%})")
print(f"emb_a != emb_b: {len(not_collapsed)} ({len(not_collapsed)/len(nonaug):.1%})")
print()

# does collapse correlate with specific programs/functions, or is it spread out?
collapsed_funcs = set(e["function"] for e in collapsed)
collapsed_samples = set(e["sample_index"] for e in collapsed)
print(f"Distinct functions among collapsed: {len(collapsed_funcs)}")
print(f"Distinct programs among collapsed: {len(collapsed_samples)}")
print()

# label balance check: does collapse correlate with a particular label?
collapsed_labels = defaultdict(int)
for e in collapsed:
    collapsed_labels[e["y_label"]] += 1
print(f"Label distribution within collapsed examples: {dict(collapsed_labels)}")

# same check on the non-collapsed set for comparison
not_collapsed_labels = defaultdict(int)
for e in not_collapsed:
    not_collapsed_labels[e["y_label"]] += 1
print(f"Label distribution within non-collapsed examples: {dict(not_collapsed_labels)}")
