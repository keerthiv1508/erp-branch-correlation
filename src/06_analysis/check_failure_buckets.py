#!/usr/bin/env python3
import sys, os, json, random, re
sys.path.insert(0, "/home_hdd/keerthivasan/exebench")
sys.path.insert(0, os.path.expanduser("~"))
import scale_pipeline as sp
from collections import defaultdict
from datasets import load_dataset

random.seed(42)
dataset = load_dataset('jordiae/exebench', split='train_real_simple_io', trust_remote_code=True)
total = len(dataset)

SAMPLE_SIZE = 3000
sample_indices = sorted(random.sample(range(total), min(SAMPLE_SIZE, total)))

compile_fail_msgs, link_fail_msgs = [], []
for i, idx in enumerate(sample_indices):
    row = dataset[idx]
    status = sp.process_sample(idx, row)
    if status.startswith('compile_fail'):
        compile_fail_msgs.append(status)
    elif status.startswith('link_fail'):
        link_fail_msgs.append(status)
    if (i + 1) % 500 == 0:
        print(f"  processed {i+1}/{len(sample_indices)}")

def bucket(msgs, patterns):
    counts = defaultdict(int)
    for m in msgs:
        hit = next((name for name, pat in patterns.items() if re.search(pat, m)), 'other')
        counts[hit] += 1
    return counts

compile_patterns = {
    "missing_bool_stdbool": r"unknown type name 'bool'",
    "implicit_declaration": r"implicit declaration of function",
    "unknown_type": r"unknown type name",
    "undeclared_identifier": r"use of undeclared identifier",
    "redefinition": r"redefinition of",
}
link_patterns = {
    "nlohmann_json_header": r"nlohmann",
    "undefined_reference": r"undefined reference",
    "deps_c_conflict": r"_deps\.c",
}

cb = bucket(compile_fail_msgs, compile_patterns)
lb = bucket(link_fail_msgs, link_patterns)

print(f"\ncompile_fail total: {len(compile_fail_msgs)}")
for k, v in sorted(cb.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/max(len(compile_fail_msgs),1):.1%})")

print(f"\nlink_fail total: {len(link_fail_msgs)}")
for k, v in sorted(lb.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/max(len(link_fail_msgs),1):.1%})")

with open(os.path.expanduser("~/failure_bucket_detail.json"), "w") as f:
    json.dump({"compile_fail_messages": compile_fail_msgs, "link_fail_messages": link_fail_msgs,
               "compile_buckets": dict(cb), "link_buckets": dict(lb)}, f, indent=2)
print("\nSaved to ~/failure_bucket_detail.json")
