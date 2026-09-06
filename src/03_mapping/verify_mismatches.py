#!/usr/bin/env python3
import re

cases = [
    ("11617", "tetris6", "!56", "!73"),
    ("38424", "unqliteSanityzeFlag", "!19", "!26"),
    ("930", "conditionals", "!21", "!34"),
    ("33827", "sumOfMutliples", "!29", "!37"),
]

TRACE_DIR = "/home_alt/keerthivasan/clang_pipeline_test"

for idx, fname, dbg_a, dbg_b in cases:
    print(f"=== sample={idx} function={fname} ({dbg_a} vs {dbg_b}) ===")
    ll_path = f"{TRACE_DIR}/{idx}_{fname}_func.ll"
    with open(ll_path) as f:
        lines = f.readlines()
    for label, dbg in [("a", dbg_a), ("b", dbg_b)]:
        num = dbg.lstrip("!")
        pattern = re.compile(rf'!dbg !{num}(?!\d)')
        matches = [l.strip() for l in lines if pattern.search(l)]
        print(f"  {label} ({dbg}): {len(matches)} matching line(s) in .ll file")
        for m in matches:
            print(f"      {m}")
    print()
