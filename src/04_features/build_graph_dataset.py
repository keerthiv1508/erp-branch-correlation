#!/usr/bin/env python3
import json
import os
import re
import subprocess
from collections import defaultdict

TRACE_DIR = os.path.expanduser("~/clang_pipeline_test")
EMBEDDED_DIR = os.path.expanduser("~/mapping_json_embedded")
PAIRS_PATH = os.path.expanduser("~/pairs_multirun.json")
OUTPUT_PATH = os.path.expanduser("~/graph_dataset.json")
DOT_CACHE_DIR = os.path.expanduser("~/cfg_dots")
CORRELATION_THRESHOLD = 0.5

os.makedirs(DOT_CACHE_DIR, exist_ok=True)

DBG_LINE_RE = re.compile(r'^\s*(.*?)\s*\[\s*(.*?)\s*\]\s*$')
DBG_TAG_RE = re.compile(r'!dbg\s+(!\d+)')
NODE_RE = re.compile(r'(Node0x[0-9a-f]+)\s*\[.*?label="(\{.*?\})"\];', re.DOTALL)
EDGE_RE = re.compile(r'(Node0x[0-9a-f]+)(?::s\d)?\s*->\s*(Node0x[0-9a-f]+);')


def parse_embeddings(embed_path):
    dbg_to_vec = {}
    dbg_is_branch = {}
    with open(embed_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("Function:"):
                continue
            m = DBG_LINE_RE.match(line)
            if not m:
                continue
            instr_text, vec_text = m.groups()
            dbg_match = DBG_TAG_RE.search(instr_text)
            if not dbg_match:
                continue
            dbg_id = dbg_match.group(1)
            try:
                vec = [float(x) for x in vec_text.split()]
            except ValueError:
                continue
            is_branch_instr = instr_text.strip().startswith("br ") or instr_text.strip().startswith("br,")
            if dbg_id not in dbg_to_vec or (is_branch_instr and not dbg_is_branch.get(dbg_id, False)):
                dbg_to_vec[dbg_id] = vec
                dbg_is_branch[dbg_id] = is_branch_instr
    return dbg_to_vec


def parse_cfg_dot(dot_path):
    with open(dot_path) as f:
        content = f.read()
    blocks = {}
    for match in NODE_RE.finditer(content):
        node_id, label = match.groups()
        blocks[node_id] = DBG_TAG_RE.findall(label)
    edges = EDGE_RE.findall(content)
    return blocks, edges


def build_function_graph(idx, fname):
    ll_path = os.path.join(TRACE_DIR, f"{idx}_{fname}_func.ll")
    embed_path = os.path.join(TRACE_DIR, f"{idx}_{fname}_embeddings.txt")
    if not os.path.exists(ll_path) or not os.path.exists(embed_path):
        return None, "missing_ll_or_embeddings"

    r = subprocess.run(["/opt/llvm-22/bin/opt", "-passes=dot-cfg", ll_path, "-disable-output"],
                        capture_output=True, text=True, cwd=DOT_CACHE_DIR, timeout=15)
    dot_path = os.path.join(DOT_CACHE_DIR, f".{fname}.dot")
    if not os.path.exists(dot_path):
        return None, "dot_cfg_failed"

    dbg_to_vec = parse_embeddings(embed_path)
    blocks, block_edges = parse_cfg_dot(dot_path)
    if not blocks:
        return None, "no_blocks_parsed"

    node_list = []
    node_embedding = []
    block_first_node = {}
    block_last_node = {}
    dbgid_to_block = {}

    for block_id, dbg_ids in blocks.items():
        indices_this_block = []
        for dbg_id in dbg_ids:
            node_idx = len(node_list)
            node_list.append(dbg_id)
            node_embedding.append(dbg_to_vec.get(dbg_id))
            indices_this_block.append(node_idx)
            dbgid_to_block[dbg_id] = block_id
        if indices_this_block:
            block_first_node[block_id] = indices_this_block[0]
            block_last_node[block_id] = indices_this_block[-1]

    n_missing = sum(1 for e in node_embedding if e is None)
    if n_missing:
        # Substitute a neutral zero-vector for nodes with no embedding (mostly phi instructions --
        # SSA merge bookkeeping, not real computation). Preserves graph topology instead of
        # discarding the whole function, which was introducing systematic bias against
        # functions with more control-flow convergence (short-circuit booleans, loop merges).
        embed_dim = next((len(e) for e in node_embedding if e is not None), 75)
        node_embedding = [e if e is not None else [0.0] * embed_dim for e in node_embedding]

    edges = []
    for block_id, dbg_ids in blocks.items():
        prev = None
        count = 0
        start = None
        for i, node_idx in enumerate(range(len(node_list))):
            pass
    # rebuild sequential edges cleanly using block grouping order captured above
    edges = []
    cursor = 0
    for block_id, dbg_ids in blocks.items():
        n = len(dbg_ids)
        block_node_indices = list(range(cursor, cursor + n))
        cursor += n
        for a, b in zip(block_node_indices, block_node_indices[1:]):
            edges.append((a, b))
    for src_block, dst_block in block_edges:
        src_last = block_last_node.get(src_block)
        dst_first = block_first_node.get(dst_block)
        if src_last is not None and dst_first is not None:
            edges.append((src_last, dst_first))

    return {
        "node_embeddings": node_embedding,
        "edges": edges,
        "dbgid_to_block": dbgid_to_block,
        "block_last_node": block_last_node,
    }, "ok"


def main():
    with open(PAIRS_PATH) as f:
        all_pairs = json.load(f)

    pairs_by_function = defaultdict(list)
    for p in all_pairs:
        pairs_by_function[(p["sample_index"], p["function"])].append(p)

    print(f"Total (sample, function) groups with pairs: {len(pairs_by_function)}")

    graph_dataset = []
    status_counts = defaultdict(int)

    for i, ((idx, fname), pairs) in enumerate(pairs_by_function.items()):
        graph_info, status = build_function_graph(idx, fname)
        status_counts[status] += 1
        if graph_info is None:
            continue

        embed_path_json = os.path.join(EMBEDDED_DIR, f"sample_{idx}.json")
        if not os.path.exists(embed_path_json):
            status_counts["missing_mapping_json"] += 1
            continue
        with open(embed_path_json) as f:
            mapping = json.load(f)
        branch_dbgid = {b["ir_branch_index"]: b.get("dbg_id") for b in mapping["branches"]}

        pair_entries = []
        for p in pairs:
            dbg_a = branch_dbgid.get(p["branch_a"]["ir_branch_index"])
            dbg_b = branch_dbgid.get(p["branch_b"]["ir_branch_index"])
            block_a = graph_info["dbgid_to_block"].get(dbg_a)
            block_b = graph_info["dbgid_to_block"].get(dbg_b)
            node_a = graph_info["block_last_node"].get(block_a) if block_a else None
            node_b = graph_info["block_last_node"].get(block_b) if block_b else None
            if node_a is None or node_b is None:
                continue
            r = p["pearson_correlation"]
            label = 1 if abs(r) >= CORRELATION_THRESHOLD else 0
            pair_entries.append({
                "node_a": node_a, "node_b": node_b,
                "y_correlation": r, "y_label": label,
            })

        if not pair_entries:
            status_counts["no_resolvable_pairs"] += 1
            continue

        graph_dataset.append({
            "sample_index": idx,
            "function": fname,
            "node_embeddings": graph_info["node_embeddings"],
            "edges": graph_info["edges"],
            "pairs": pair_entries,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(pairs_by_function)} functions...")

    print(f"\n=== SUMMARY ===")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    print(f"\nFunctions with valid graphs: {len(graph_dataset)}")
    total_pairs = sum(len(g["pairs"]) for g in graph_dataset)
    print(f"Total branch pairs covered: {total_pairs}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(graph_dataset, f)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
