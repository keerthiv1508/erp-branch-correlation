#!/usr/bin/env python3
import glob
import json
import os
from collections import defaultdict

PAIRS_PATH = os.path.expanduser("~/pairs_multirun.json")
EMBEDDED_DIR = os.path.expanduser("~/mapping_json_embedded")
TRACE_DIR = os.path.expanduser("~/clang_pipeline_test")
CORRELATION_THRESHOLD = 0.5


def normalize_addr(addr):
    addr = addr.lower()
    if addr.startswith("0x"):
        addr = addr[2:]
    return addr.lstrip("0") or "0"


def min_distance_sorted(a_pos, b_pos):
    i, j = 0, 0
    best = None
    while i < len(a_pos) and j < len(b_pos):
        d = abs(a_pos[i] - b_pos[j])
        if best is None or d < best:
            best = d
        if a_pos[i] < b_pos[j]:
            i += 1
        else:
            j += 1
    return best


def load_mapping(sample_idx):
    path = os.path.join(EMBEDDED_DIR, f"sample_{sample_idx}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_trace_positions(sample_idx, fname):
    """Returns list of dicts (one per run), each addr -> sorted list of positions."""
    run_traces = sorted(glob.glob(os.path.join(TRACE_DIR, f"{sample_idx}_{fname}_run*_trace.out")))
    per_run = []
    for t in run_traces:
        positions = defaultdict(list)
        with open(t) as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 2:
                    positions[normalize_addr(parts[0])].append(i)
        per_run.append(positions)
    return per_run


def main():
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs")

    by_sample = defaultdict(list)
    for p in pairs:
        by_sample[p["sample_index"]].append(p)

    pair_records = []
    vector_groups = defaultdict(list)
    skipped_no_embedding = 0
    skipped_no_addr = 0

    n_samples = len(by_sample)
    for si, (idx, sample_pairs) in enumerate(by_sample.items()):
        if si % 200 == 0:
            print(f"  processing sample {si}/{n_samples}...")

        mapping = load_mapping(idx)
        if mapping is None:
            skipped_no_embedding += len(sample_pairs)
            continue
        fname = mapping["function"]

        branch_lookup = {b["ir_branch_index"]: b for b in mapping["branches"]}
        trace_positions = None  # lazy load only if needed

        for p in sample_pairs:
            ba = branch_lookup.get(p["branch_a"]["ir_branch_index"])
            bb = branch_lookup.get(p["branch_b"]["ir_branch_index"])
            if ba is None or bb is None or not ba.get("embedding") or not bb.get("embedding"):
                skipped_no_embedding += 1
                continue

            emb_a, emb_b = ba["embedding"], bb["embedding"]
            x_fwd = tuple(round(v, 6) for v in emb_a + emb_b)
            x_rev = tuple(round(v, 6) for v in emb_b + emb_a)

            r = p["pearson_correlation"]
            label = 1 if abs(r) >= CORRELATION_THRESHOLD else 0
            vector_groups[x_fwd].append(label)
            vector_groups[x_rev].append(label)

            if not ba.get("region_first_addresses") or not bb.get("region_first_addresses"):
                skipped_no_addr += 1
                dist = None
            else:
                if trace_positions is None:
                    trace_positions = load_trace_positions(idx, fname)
                addr_a = normalize_addr(ba["region_first_addresses"][0])
                addr_b = normalize_addr(bb["region_first_addresses"][0])
                run_dists = []
                for positions in trace_positions:
                    pa, pb = positions.get(addr_a), positions.get(addr_b)
                    if pa and pb:
                        run_dists.append(min_distance_sorted(sorted(pa), sorted(pb)))
                dist = sorted(run_dists)[len(run_dists) // 2] if run_dists else None  # median

            pair_records.append({"x_fwd": x_fwd, "label": label, "distance": dist})

    print(f"\nBuilt {len(pair_records)} pair records "
          f"({skipped_no_embedding} skipped missing embedding, {skipped_no_addr} missing address)")

    conflicted_vectors = set()
    for key, labels in vector_groups.items():
        if 0 in labels and 1 in labels:
            conflicted_vectors.add(key)

    for rec in pair_records:
        rec["conflicted"] = rec["x_fwd"] in conflicted_vectors

    with_dist = [r for r in pair_records if r["distance"] is not None]
    print(f"Pairs with a computable distance: {len(with_dist)} / {len(pair_records)}")

    buckets = [(0, 5), (6, 20), (21, 100), (101, 1000), (1001, float("inf"))]
    print(f"\n{'Distance range':<20}{'# pairs':<10}{'# conflicted':<14}{'% conflicted':<12}")
    for lo, hi in buckets:
        in_bucket = [r for r in with_dist if lo <= r["distance"] <= hi]
        if not in_bucket:
            continue
        n_conf = sum(1 for r in in_bucket if r["conflicted"])
        label = f"{lo}-{hi if hi != float('inf') else '+'}"
        print(f"{label:<20}{len(in_bucket):<10}{n_conf:<14}{100*n_conf/len(in_bucket):.1f}%")

    conf_dists = [r["distance"] for r in with_dist if r["conflicted"]]
    clean_dists = [r["distance"] for r in with_dist if not r["conflicted"]]
    def median(xs):
        xs = sorted(xs)
        return xs[len(xs)//2] if xs else None
    print(f"\nMedian distance, conflicted pairs:     {median(conf_dists)}")
    print(f"Median distance, non-conflicted pairs: {median(clean_dists)}")

    with open("/home_alt/keerthivasan/distance_vs_conflict_output.json", "w") as f:
        json.dump({"pair_records": [{"label": r["label"], "distance": r["distance"], "conflicted": r["conflicted"]} for r in pair_records]}, f)

if __name__ == "__main__":
    main()
