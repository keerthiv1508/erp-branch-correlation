#!/usr/bin/env python3
"""
Export binary-to-IR branch mapping as structured JSON, for later joining
with Pin trace binary addresses.

Usage:
    python3 export_mapping_json.py <file.o> <file.ll> [output.json]

Reuses the core parsing/grouping/matching logic from map_branches.py
without modifying it.
"""

import sys
import os
import json
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_branches import (
    run_objdump, parse_binary_jumps, group_into_regions, parse_ll_file
)

FUNC_NAME_RE = re.compile(r'^\s*define\s+.*@(\w+)\s*\(')


def get_function_name(ll_path):
    with open(ll_path) as f:
        for line in f:
            m = FUNC_NAME_RE.match(line)
            if m:
                return m.group(1)
    return "unknown"


def build_mapping(obj_path, ll_path):
    objdump_output = run_objdump(obj_path)
    jumps = parse_binary_jumps(objdump_output)
    regions = group_into_regions(jumps)
    ir_branches = parse_ll_file(ll_path)
    function_name = get_function_name(ll_path)

    result = {
        "object_file": os.path.basename(obj_path),
        "ir_file": os.path.basename(ll_path),
        "function": function_name,
        "ir_branch_count": len(ir_branches),
        "binary_region_count": len(regions),
        "branches": []
    }

    region_idx = 0
    for idx, ir_b in enumerate(ir_branches):
        k = ir_b.num_cases
        consumed = regions[region_idx:region_idx + k]
        all_lines = sorted(set(l for r in consumed for l in r.lines))
        addrs = [r.first_addr for r in consumed]
        all_addrs = [j.address for r in consumed for j in r.jumps]
        consistent = ir_b.src_line in all_lines if ir_b.src_line else False

        result["branches"].append({
            "ir_branch_index": idx,
            "kind": ir_b.kind,
            "ir_line": ir_b.ll_line_no,
            "dbg_id": ir_b.dbg_id,
            "src_line": ir_b.src_line,
            "src_col": ir_b.src_col,
            "num_cases": ir_b.num_cases,
            "region_first_addresses": addrs,
            "all_binary_addresses": all_addrs,
            "consistent": consistent
        })
        region_idx += k

    result["unmatched_regions"] = len(regions) - region_idx
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 export_mapping_json.py <file.o> <file.ll> [output.json]")
        sys.exit(1)

    obj_path = sys.argv[1]
    ll_path = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    mapping = build_mapping(obj_path, ll_path)

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Wrote mapping to {out_path}")
    else:
        print(json.dumps(mapping, indent=2))


if __name__ == '__main__':
    main()
