#!/usr/bin/env python3
import glob
import json
import os
import re
import subprocess

MAPPING_DIR = os.path.expanduser("~/mapping_json_clang")
TRACE_DIR = os.path.expanduser("~/clang_pipeline_test")
OUTPUT_DIR = os.path.expanduser("~/mapping_json_embedded")
VOCAB_PATH = "/home_hdd/shared/llvm-project-22.1.0.src/llvm/lib/Analysis/models/seedEmbeddingVocab75D.json"
IR2VEC_BIN = "/opt/llvm-22/bin/llvm-ir2vec"

DBG_LINE_RE = re.compile(r'^\s*(.*?)\s*\[\s*(.*?)\s*\]\s*$')
DBG_TAG_RE = re.compile(r'!dbg\s+(!\d+)')


def run_ir2vec(ll_path, out_path):
    cmd = [
        IR2VEC_BIN, "embeddings", "--mode=llvm",
        f"--ir2vec-vocab-path={VOCAB_PATH}",
        "--ir2vec-kind=flow-aware",
        "--level=inst",
        ll_path, "-o", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping_files = sorted(glob.glob(os.path.join(MAPPING_DIR, "sample_*.json")))
    print(f"Found {len(mapping_files)} mapping JSONs")

    results = []
    for mpath in mapping_files:
        with open(mpath) as f:
            mapping = json.load(f)
        idx = os.path.basename(mpath).replace("sample_", "").replace(".json", "")
        fname = mapping["function"]

        ll_path = os.path.join(TRACE_DIR, f"{idx}_{fname}_func.ll")
        if not os.path.exists(ll_path):
            print(f"  Sample {idx} ({fname}): no .ll file found, skipping")
            continue

        embed_path = os.path.join(TRACE_DIR, f"{idx}_{fname}_embeddings.txt")
        ok, err = run_ir2vec(ll_path, embed_path)
        if not ok:
            print(f"  Sample {idx} ({fname}): ir2vec FAILED: {err[:200]}")
            results.append((idx, fname, 0, len(mapping["branches"]), "ir2vec_fail"))
            continue

        dbg_to_vec = parse_embeddings(embed_path)

        n_matched = 0
        for branch in mapping["branches"]:
            dbg_id = branch.get("dbg_id")
            if dbg_id and dbg_id in dbg_to_vec:
                branch["embedding"] = dbg_to_vec[dbg_id]
                n_matched += 1
            else:
                branch["embedding"] = None

        out_path = os.path.join(OUTPUT_DIR, f"sample_{idx}.json")
        with open(out_path, "w") as f:
            json.dump(mapping, f, indent=2)

        n_total = len(mapping["branches"])
        results.append((idx, fname, n_matched, n_total, "ok"))
        print(f"  Sample {idx} ({fname}): {n_matched}/{n_total} branches matched to embeddings")

    print("\n=== SUMMARY ===")
    total_matched = sum(r[2] for r in results)
    total_branches = sum(r[3] for r in results)
    print(f"{total_matched}/{total_branches} branches matched across {len(results)} samples")


if __name__ == "__main__":
    main()
