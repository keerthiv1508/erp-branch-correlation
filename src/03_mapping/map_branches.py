#!/usr/bin/env python3
"""
Map binary branch instructions to LLVM IR branch instructions for a single
compiled function, using debug info as the linking mechanism.

Usage:
    python3 map_branches.py <basename>

Expects <basename>.o (compiled with -g -O0) and <basename>.ll (compiled with
-S -emit-llvm -g -O0) to exist in the current directory, both from the same
source file.

Strategy:
  - The .ll file is treated as the structural anchor: every br/switch
    instruction there is well-defined, with a !dbg tag resolving to a
    (line, column) via !DILocation.
  - The binary side (via `objdump -d -l`) is noisier: a single high-level
    branch can lower to multiple conditional jumps (compound && / || chains),
    and a single IR `switch` can lower to a sequential run of cmp/jump pairs
    rather than a jump table (observed at -O0 for small switches).
  - Binary jumps are grouped into "logical branch regions" using raw
    instruction-stream adjacency (a jump immediately following another jump
    with no computation between them belongs to the same region) -- NOT
    source line, which is provably unreliable for chained && / || conditions.
  - IR branches are matched to binary regions by order of appearance.
  - Any count mismatch is reported explicitly, never silently forced.
"""

import re
import subprocess
import sys
from dataclasses import dataclass, field


COND_JUMP_RE = re.compile(
    r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+\s*(j(?!mp)\w+)\s'
)

ADDR_RE = re.compile(r'^\s*([0-9a-f]+):')
SRC_LINE_RE = re.compile(r':(\d+)\s*$')


@dataclass
class BinaryJump:
    address: str
    mnemonic: str
    src_line: int
    starts_new_region: bool = False


@dataclass
class BinaryRegion:
    jumps: list = field(default_factory=list)

    @property
    def lines(self):
        return sorted(set(j.src_line for j in self.jumps))

    @property
    def first_addr(self):
        return self.jumps[0].address if self.jumps else None


@dataclass
class IRBranch:
    ll_line_no: int
    kind: str
    dbg_id: str
    src_line: int = None
    src_col: int = None
    num_cases: int = 1


def run_objdump(obj_path):
    result = subprocess.run(
        ['/opt/llvm-22/bin/llvm-objdump', '-d', '-l', obj_path],
        capture_output=True, text=True, check=True
    )
    return result.stdout


def parse_binary_jumps(objdump_output):
    jumps = []
    current_line = None
    prev_was_jump = False
    for raw_line in objdump_output.splitlines():
        stripped = raw_line.strip()

        if stripped and not ADDR_RE.match(raw_line) and re.search(r'\.c:\d+$', stripped):
            m = SRC_LINE_RE.search(stripped)
            if m:
                current_line = int(m.group(1))
            continue

        if not ADDR_RE.match(raw_line):
            continue

        cond_match = COND_JUMP_RE.match(raw_line)
        if cond_match:
            addr_match = ADDR_RE.match(raw_line)
            addr = addr_match.group(1) if addr_match else '?'
            mnemonic = cond_match.group(1)
            if current_line is not None:
                jumps.append(BinaryJump(
                    address=addr, mnemonic=mnemonic, src_line=current_line,
                    starts_new_region=not prev_was_jump
                ))
            prev_was_jump = True
        else:
            prev_was_jump = False
    return jumps


def group_into_regions(jumps):
    if not jumps:
        return []
    regions = []
    current = BinaryRegion(jumps=[jumps[0]])
    for j in jumps[1:]:
        if j.starts_new_region:
            regions.append(current)
            current = BinaryRegion(jumps=[j])
        else:
            current.jumps.append(j)
    regions.append(current)
    return regions

def compute_expected_switch_regions(case_arms):
    from collections import defaultdict
    groups = defaultdict(list)
    for val, label in case_arms:
        groups[label].append(val)
    total_regions = 0
    for label, vals in groups.items():
        vals = sorted(set(vals))
        runs = 1
        for i in range(1, len(vals)):
            if vals[i] != vals[i - 1] + 1:
                runs += 1
        total_regions += runs
    return total_regions

def parse_ll_file(ll_path):
    with open(ll_path) as f:
        text = f.read()
    lines = text.splitlines()

    diloc = {}
    diloc_re = re.compile(r'^(!\d+)\s*=\s*!DILocation\(line:\s*(\d+),\s*column:\s*(\d+)')
    for raw in lines:
        m = diloc_re.match(raw)
        if m:
            diloc[m.group(1)] = (int(m.group(2)), int(m.group(3)))

    branches = []
    br_re = re.compile(r'\bbr\s+i1\b.*!dbg\s+(!\d+)')
    CASE_ARM_RE = re.compile(r'^\s*i\d+\s+(-?\d+),\s*label\s+%(\S+)')
    switch_start_re = re.compile(r'^\s*switch\s+i\d+\s')
    switch_end_re = re.compile(r'^\s*\]\s*,\s*!dbg\s+(!\d+)')

    i = 0
    while i < len(lines):
        raw = lines[i]
        if switch_start_re.match(raw):
            start_i = i
            dbg_id = None
            j = i
            case_arms = []
            while j < len(lines):
                if j>start_i:
                   cm = CASE_ARM_RE.match(lines[j])
                   if cm:
                       case_arms.append((int(cm.group(1)), cm.group(2)))
                m = switch_end_re.match(lines[j])
                if m:
                    dbg_id = m.group(1)
                    break
                j += 1
            if dbg_id:
                line, col = diloc.get(dbg_id, (None, None))
                num_cases = compute_expected_switch_regions(case_arms)
                branches.append(IRBranch(ll_line_no=start_i + 1, kind='switch',  dbg_id=dbg_id, src_line=line, src_col=col,num_cases=num_cases))
            i = j + 1
            continue

        m = br_re.search(raw)
        if m:
            dbg_id = m.group(1)
            line, col = diloc.get(dbg_id, (None, None))
            branches.append(IRBranch(ll_line_no=i + 1, kind='br', dbg_id=dbg_id,
                                      src_line=line, src_col=col))
        i += 1
    return branches

def match(ir_branches, regions):
    total_expected = sum(b.num_cases for b in ir_branches)
    print(f"IR branch instructions (conditional only): {len(ir_branches)}")
    print(f"IR branches expanded for switch arms:      {total_expected}")
    print(f"Binary logical branch regions:              {len(regions)}")
    print()

    if total_expected != len(regions):
        print("!! COUNT MISMATCH - ordered matching below is not guaranteed reliable.")
        print("   Inspect manually before trusting this mapping.\n")

    print(f"{'#':<3} {'IR line':<8} {'Kind':<7} {'IR src':<10} {'Regions (addr)':<20} {'Consistent?'}")
    print("-" * 75)

    region_idx = 0
    unmatched_ir = 0
    for idx, ir_b in enumerate(ir_branches):
        k = ir_b.num_cases
        consumed = regions[region_idx:region_idx + k]
        if len(consumed) < k:
            print(f"{idx:<3} {ir_b.ll_line_no:<8} {ir_b.kind:<7} {'?':<10} not enough regions left    CHECK")
            unmatched_ir += 1
            region_idx += len(consumed)
            continue
        ir_src = f"{ir_b.src_line}:{ir_b.src_col}" if ir_b.src_line else "?"
        all_lines = sorted(set(l for r in consumed for l in r.lines))
        addrs = ",".join(r.first_addr for r in consumed)
        consistent = ir_b.src_line in all_lines if ir_b.src_line else False
        flag = "OK" if consistent else "CHECK"
        print(f"{idx:<3} {ir_b.ll_line_no:<8} {ir_b.kind:<7} {ir_src:<10} {addrs:<20} {flag}")
        region_idx += k

    if region_idx < len(regions):
        print(f"\n{len(regions) - region_idx} unmatched binary region(s) (no corresponding IR branch found)")
    if unmatched_ir:
        print(f"{unmatched_ir} IR branch(es) could not be matched (ran out of binary regions)")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 map_branches.py <file.o> <file.ll>")
        sys.exit(1)

    obj_path = sys.argv[1]
    ll_path = sys.argv[2]

    objdump_output = run_objdump(obj_path)
    jumps = parse_binary_jumps(objdump_output)
    regions = group_into_regions(jumps)

    ir_branches = parse_ll_file(ll_path)

    match(ir_branches, regions)


if __name__ == '__main__':
    main()
