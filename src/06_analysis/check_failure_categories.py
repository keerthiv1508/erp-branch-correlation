#!/usr/bin/env python3
import sys, os, json, random
sys.path.insert(0, "/home_hdd/keerthivasan/exebench")
sys.path.insert(0, os.path.expanduser("~"))
import scale_pipeline as sp
from collections import defaultdict
from datasets import load_dataset

random.seed(42)

dataset = load_dataset('jordiae/exebench', split='train_real_simple_io', trust_remote_code=True)
total = len(dataset)
print(f"Total dataset size: {total}")

SAMPLE_SIZE = 3000
sample_indices = sorted(random.sample(range(total), min(SAMPLE_SIZE, total)))

category_counts = defaultdict(int)
category_examples = defaultdict(list)

for i, idx in enumerate(sample_indices):
    row = dataset[idx]
    status = sp.process_sample(idx, row)
    cat = status.split(':')[0].strip()
    category_counts[cat] += 1
    if len(category_examples[cat]) < 3 and ':' in status:
        category_examples[cat].append(status[:250])
    if (i + 1) % 200 == 0:
        print(f"  processed {i+1}/{len(sample_indices)}")

print("\n=== CATEGORY BREAKDOWN (sampled) ===")
for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    pct = count / len(sample_indices) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

out = {
    "sample_size": len(sample_indices),
    "total_dataset": total,
    "category_counts": dict(category_counts),
    "category_examples": dict(category_examples),
}
with open(os.path.expanduser("~/failure_categories_sample.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to ~/failure_categories_sample.json")
