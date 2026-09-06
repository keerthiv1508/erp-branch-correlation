#!/usr/bin/env python3
import json, os
from collections import defaultdict

PAIRS_PATH = os.path.expanduser("~/pairs_multirun.json")

with open(PAIRS_PATH) as f:
    pairs = json.load(f)

distinct_samples = set()
distinct_functions = set()
func_to_samples = defaultdict(set)

for p in pairs:
    idx = p["sample_index"]
    fname = p["function"]
    distinct_samples.add(idx)
    distinct_functions.add(fname)
    func_to_samples[fname].add(idx)

print(f"Total pairs: {len(pairs)}")
print(f"Distinct programs (sample_index): {len(distinct_samples)}")
print(f"Distinct function names: {len(distinct_functions)}")
print()

shared = {fn: samples for fn, samples in func_to_samples.items() if len(samples) > 1}
print(f"Function names appearing in >1 program: {len(shared)} of {len(distinct_functions)}")
print(f"Sum of (occurrences - 1) across shared names: {sum(len(s) - 1 for s in shared.values())}")
print(f"  (this should roughly equal: distinct_programs - distinct_functions = "
      f"{len(distinct_samples) - len(distinct_functions)})")
print()

top = sorted(shared.items(), key=lambda kv: -len(kv[1]))[:20]
print("Top 20 most-shared function names:")
for fn, samples in top:
    print(f"  {fn!r}: appears in {len(samples)} distinct programs")
