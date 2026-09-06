#!/usr/bin/env python3
"""
Batch-test map_branches.py's core functions across multiple ExeBench samples.
Compiles each .c to .o and .ll, runs the mapping logic, and classifies the
result as CLEAN (counts match, all consistent), MISMATCH (count mismatch),
INCONSISTENT (counts match but some CHECK flags), or COMPILE_FAIL / NO_BRANCHES.
"""

import subprocess
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_branches import (
    run_objdump, parse_binary_jumps, group_into_regions, parse_ll_file
)

SAMPLE_DIR = "/home_alt/keerthivasan/exebench_samples"
WORKDIR = "/tmp/mapping_batch"
os.makedirs(WORKDIR, exist_ok=True)


def compile_pair(c_path, tag):
    obj_path = os.path.join(WORKDIR, f"{tag}.o")
    ll_path = os.path.join(WORKDIR, f"{tag}.ll")
    r1 = subprocess.run(["clang", "-g", "-O0", "-c", c_path, "-o", obj_path],
                         capture_output=True, text=True)
    if r1.returncode != 0:
        return None, None, r1.stderr.strip().splitlines()[-1] if r1.stderr else "unknown error"
    r2 = subprocess.run(["clang", "-S", "-emit-llvm", "-g", "-O0", c_path, "-o", ll_path],
                         capture_output=True, text=True)
    if r2.returncode != 0:
        return None, None, r2.stderr.strip().splitlines()[-1] if r2.stderr else "unknown error"
    return obj_path, ll_path, None


def evaluate(obj_path, ll_path):
    try:
        objdump_output = run_objdump(obj_path)
        jumps = parse_binary_jumps(objdump_output)
        regions = group_into_regions(jumps)
        ir_branches = parse_ll_file(ll_path)
    except Exception as e:
        return "SCRIPT_ERROR", str(e)

    if not ir_branches:
        return "NO_BRANCHES", f"0 IR branches, {len(regions)} binary regions"

    total_expected = sum(b.num_cases for b in ir_branches)

    if total_expected != len(regions):
        return "MISMATCH", f"{total_expected} expected vs {len(regions)} regions"

    region_idx = 0
    all_ok = True
    for ir_b in ir_branches:
        k = ir_b.num_cases
        consumed = regions[region_idx:region_idx + k]
        all_lines = sorted(set(l for r in consumed for l in r.lines))
        if ir_b.src_line not in all_lines:
            all_ok = False
        region_idx += k

    if all_ok:
        return "CLEAN", f"{len(ir_branches)} IR branches, {len(regions)} regions, all consistent"
    else:
        return "INCONSISTENT", f"{len(ir_branches)} IR branches, {len(regions)} regions, some CHECK flags"


def main():
    files = sorted(
        glob.glob(os.path.join(SAMPLE_DIR, "sample_*.c")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])
    )
    results = []
    for c_path in files:
        i = int(os.path.basename(c_path).split("_")[1].split(".")[0])
        tag = f"sample_{i}"
        obj_path, ll_path, err = compile_pair(c_path, tag)
        if err:
            results.append((i, "COMPILE_FAIL", err))
            continue
        status, detail = evaluate(obj_path, ll_path)
        results.append((i, status, detail))

    print(f"{'#':<4} {'Status':<14} {'Detail'}")
    print("-" * 80)
    for i, status, detail in results:
        print(f"{i:<4} {status:<14} {detail}")

    print()
    from collections import Counter
    counts = Counter(r[1] for r in results)
    print("Summary:", dict(counts))


if __name__ == '__main__':
    main()
