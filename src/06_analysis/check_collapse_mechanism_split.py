#!/usr/bin/env python3
import json, os
from collections import defaultdict
import numpy as np

DATA_PATH = os.path.expanduser("~/training_data.json")
PAIRS_PATH = os.path.expanduser("~/pairs_multirun.json")
MAPPING_DIR = os.path.expanduser("~/mapping_json_clang")

with open(DATA_PATH) as f:
    data = json.load(f)
with open(PAIRS_PATH) as f:
    pairs = json.load(f)

pair_lookup = defaultdict(list)
for p in pairs:
    key = (p["sample_index"], p["function"], round(p["pearson_correlation"], 6))
    pair_lookup[key].append(p)

mapping_cache = {}
def get_dbg_id(sample_index, ir_branch_index):
    if sample_index not in mapping_cache:
        path = os.path.join(MAPPING_DIR, f"sample_{sample_index}.json")
        if not os.path.exists(path):
            mapping_cache[sample_index] = {}
        else:
            with open(path) as f:
                m = json.load(f)
            mapping_cache[sample_index] = {b["ir_branch_index"]: b.get("dbg_id") for b in m["branches"]}
    return mapping_cache[sample_index].get(ir_branch_index)

collapsed = []
for e in data:
    if e.get("augmented"):
        continue
    x = np.array(e["x"])
    if np.allclose(x[:75], x[75:]):
        collapsed.append(e)

same_dbgid = 0       # mechanism A: dbg_a == dbg_b, forced identical by construction
diff_dbgid = 0       # mechanism B: dbg_a != dbg_b, embedding coarseness
no_match = 0

for e in collapsed:
    k = (e["sample_index"], e["function"], round(e["y_correlation"], 6))
    m = pair_lookup.get(k, [])
    if not m:
        no_match += 1
        continue
    p = m[0]
    idx = e["sample_index"]
    dbg_a = get_dbg_id(idx, p["branch_a"]["ir_branch_index"])
    dbg_b = get_dbg_id(idx, p["branch_b"]["ir_branch_index"])
    if not dbg_a or not dbg_b:
        no_match += 1
        continue
    if dbg_a == dbg_b:
        same_dbgid += 1
    else:
        diff_dbgid += 1

total = len(collapsed)
print(f"Total collapsed pairs (emb_a==emb_b): {total}")
print(f"Mechanism A - same dbg_id within pair (forced by construction): {same_dbgid} ({same_dbgid/total:.1%})")
print(f"Mechanism B - different dbg_id, coincidentally equal vector: {diff_dbgid} ({diff_dbgid/total:.1%})")
print(f"No pair match found: {no_match} ({no_match/total:.1%})")
