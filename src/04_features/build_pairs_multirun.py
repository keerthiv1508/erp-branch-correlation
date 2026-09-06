#!/usr/bin/env python3
import glob
import json
import math
import os

MAPPING_DIR = os.path.expanduser("~/mapping_json_clang")
TRACE_DIR = os.path.expanduser("~/clang_pipeline_test")
OUTPUT_PATH = os.path.expanduser("~/pairs_multirun.json")
MIN_OVERLAP = 3


def normalize_addr(addr):
    addr = addr.lower()
    if addr.startswith("0x"):
        addr = addr[2:]
    return addr.lstrip("0") or "0"


def compute_branch_ratio_in_trace(trace_path, target_addr):
    taken = 0
    total = 0
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            addr, outcome = parts
            if normalize_addr(addr) != target_addr:
                continue
            total += 1
            if outcome == "taken":
                taken += 1
    return taken / total if total > 0 else None


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None
    return cov / denom


def main():
    mapping_files = sorted(glob.glob(os.path.join(MAPPING_DIR, "sample_*.json")))
    print(f"Found {len(mapping_files)} mapping JSONs")

    all_pairs = []
    total_tested = 0
    total_reliable = 0

    for mpath in mapping_files:
        with open(mpath) as f:
            mapping = json.load(f)
        idx = os.path.basename(mpath).replace("sample_", "").replace(".json", "")
        fname = mapping["function"]

        run_traces = sorted(glob.glob(os.path.join(TRACE_DIR, f"{idx}_{fname}_run*_trace.out")))
        if not run_traces:
            print(f"  Sample {idx} ({fname}): no multi-run traces found, skipping")
            continue

        branches = mapping["branches"]
        branches = [b for b in branches if b.get("region_first_addresses")]

        branch_ratios = {}
        for b in branches:
            addr = normalize_addr(b["region_first_addresses"][0])
            ratios = [compute_branch_ratio_in_trace(t, addr) for t in run_traces]
            branch_ratios[b["ir_branch_index"]] = ratios

        n_scorable = 0
        n_tested = 0
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                bi, bj = branches[i], branches[j]
                ri = branch_ratios[bi["ir_branch_index"]]
                rj = branch_ratios[bj["ir_branch_index"]]
                xs, ys = [], []
                for a, b in zip(ri, rj):
                    if a is not None and b is not None:
                        xs.append(a)
                        ys.append(b)
                if len(xs) < MIN_OVERLAP:
                    continue
                n_tested += 1
                corr = pearson(xs, ys)
                if corr is None:
                    continue
                n_scorable += 1
                all_pairs.append({
                    "sample_index": idx,
                    "function": fname,
                    "branch_a": {"ir_branch_index": bi["ir_branch_index"], "src_line": bi.get("src_line")},
                    "branch_b": {"ir_branch_index": bj["ir_branch_index"], "src_line": bj.get("src_line")},
                    "n_overlapping_runs": len(xs),
                    "pearson_correlation": corr,
                })

        total_tested += n_tested
        total_reliable += n_scorable
        print(f"  Sample {idx} ({fname}): {len(branches)} branches, {len(run_traces)} runs, "
              f"{n_tested} pairs tested (>= {MIN_OVERLAP} overlap), {n_scorable} scorable")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\n{len(all_pairs)} scorable pairs written to {OUTPUT_PATH}")
    if total_tested:
        print(f"Overall: {total_reliable}/{total_tested} scorable ({100*total_reliable/total_tested:.1f}%)")

    all_pairs.sort(key=lambda p: abs(p["pearson_correlation"]), reverse=True)
    print("\nTop 10 strongest correlations:")
    for p in all_pairs[:10]:
        print(f"  {p['function']} (sample {p['sample_index']}): "
              f"line{p['branch_a']['src_line']} vs line{p['branch_b']['src_line']} "
              f"-> r={p['pearson_correlation']:.3f} (n={p['n_overlapping_runs']} runs)")


if __name__ == "__main__":
    main()
