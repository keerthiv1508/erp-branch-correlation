#!/usr/bin/env python3
"""
Batch-export binary-to-IR branch mappings as JSON for all ExeBench samples
that contain at least one branch. Skips samples with no branches or that
fail to compile, logging both.

Usage:
    python3 export_mapping_json_batch.py [output_dir]
"""

import sys
import os
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mapping_batch import compile_pair, SAMPLE_DIR
from export_mapping_json import build_mapping

OUT_DIR_DEFAULT = "/home_alt/keerthivasan/mapping_json"


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR_DEFAULT
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(SAMPLE_DIR, "sample_*.c")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])
    )

    written = 0
    no_branches = 0
    compile_fail = 0
    errors = 0

    for c_path in files:
        i = int(os.path.basename(c_path).split("_")[1].split(".")[0])
        tag = f"sample_{i}"
        obj_path, ll_path, err = compile_pair(c_path, tag)
        if err:
            compile_fail += 1
            continue

        try:
            mapping = build_mapping(obj_path, ll_path)
        except Exception as e:
            print(f"sample_{i}: ERROR building mapping - {e}")
            errors += 1
            continue

        if mapping["ir_branch_count"] == 0:
            no_branches += 1
            continue

        mapping["sample_index"] = i
        out_path = os.path.join(out_dir, f"sample_{i}.json")
        with open(out_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        written += 1

    print(f"\nDone.")
    print(f"  JSON files written: {written}  (in {out_dir})")
    print(f"  No branches (skipped): {no_branches}")
    print(f"  Compile failures (skipped): {compile_fail}")
    print(f"  Mapping errors: {errors}")


if __name__ == '__main__':
    main()
