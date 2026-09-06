#!/usr/bin/env python3
import json
import os

PAIRS_PATH = os.path.expanduser("~/pairs_multirun.json")
EMBEDDED_DIR = os.path.expanduser("~/mapping_json_embedded")
OUTPUT_PATH = os.path.expanduser("~/training_data.json")
CORRELATION_THRESHOLD = 0.5


def load_embedded_mapping(sample_idx):
    path = os.path.join(EMBEDDED_DIR, f"sample_{sample_idx}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_branch_embedding(mapping, ir_branch_index):
    for b in mapping["branches"]:
        if b["ir_branch_index"] == ir_branch_index:
            return b.get("embedding")
    return None


def main():
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} correlation pairs")

    examples = []
    skipped = 0

    for pair in pairs:
        idx = pair["sample_index"]
        mapping = load_embedded_mapping(idx)
        if mapping is None:
            skipped += 1
            continue

        emb_a = get_branch_embedding(mapping, pair["branch_a"]["ir_branch_index"])
        emb_b = get_branch_embedding(mapping, pair["branch_b"]["ir_branch_index"])
        if emb_a is None or emb_b is None:
            skipped += 1
            continue

        r = pair["pearson_correlation"]
        label = 1 if abs(r) >= CORRELATION_THRESHOLD else 0

        examples.append({
            "sample_index": idx,
            "function": pair["function"],
            "x": emb_a + emb_b,
            "y_correlation": r,
            "y_label": label,
            "augmented": False,
        })
        examples.append({
            "sample_index": idx,
            "function": pair["function"],
            "x": emb_b + emb_a,
            "y_correlation": r,
            "y_label": label,
            "augmented": True,
        })

    print(f"Built {len(examples)} training examples ({skipped} pairs skipped -- missing embeddings)")
    print(f"Input dimension: {len(examples[0]['x']) if examples else 'N/A'}")
    n_positive = sum(1 for e in examples if e["y_label"] == 1)
    print(f"Label balance: {n_positive}/{len(examples)} positive (|r| >= {CORRELATION_THRESHOLD})")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(examples, f)
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
