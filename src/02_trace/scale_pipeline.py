#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import subprocess

from datasets import load_dataset
from exebench import exebench_dict_to_dict

OUTDIR = os.path.expanduser("~/clang_pipeline_test")
MAPPING_DIR = os.path.expanduser("~/mapping_json_clang")
EXEBENCH_PKG_DIR = "/home_hdd/keerthivasan/exebench/exebench"
PIN_ROOT = '/home_alt/keerthivasan/software/pin-3.28-98749-g6643ecee5-gcc-linux'
PIN_TOOL = f'{PIN_ROOT}/source/tools/ManualExamples/obj-intel64/branch_trace.so'
MAPPING_SCRIPT = "/home_alt/keerthivasan/export_mapping_json.py"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MAPPING_DIR, exist_ok=True)


def strip_linkage_keywords(func_def):
    return re.sub(r'^\s*(?:(?:static|inline)\s+)+', '', func_def)


COMMON_HEADER_PREFIX = (
    "#include <limits.h>\n"
    "#include <float.h>\n"
    "#include <math.h>\n"
    "#include <string.h>\n"
    "#include <stdlib.h>\n"
    "#include <ctype.h>\n"
    "#include <stdint.h>\n"
)


def prepare_func_source(func_def):
    return COMMON_HEADER_PREFIX + "#line 1\n" + strip_linkage_keywords(func_def)


def process_sample(idx, row):
    fname = row['fname']
    if len(row['func_def']) < 100:
        return 'skip_trivial'

    exe_path = os.path.join(OUTDIR, f"{idx}_{fname}_clang_exe")
    mapping_path = os.path.join(MAPPING_DIR, f"sample_{idx}.json")
    if os.path.exists(exe_path) and os.path.exists(mapping_path):
        return 'already_done'

    try:
        func_c_path = os.path.join(OUTDIR, f"{idx}_{fname}_func.c")
        func_obj_path = os.path.join(OUTDIR, f"{idx}_{fname}_func.o")
        func_ll_path = os.path.join(OUTDIR, f"{idx}_{fname}_func.ll")
        func_def_fixed = prepare_func_source(row['func_def'])
        with open(func_c_path, "w") as f:
            f.write(func_def_fixed)

        r1 = subprocess.run(["clang", "-g", "-O0", "-c", func_c_path, "-o", func_obj_path],
                             capture_output=True, text=True, timeout=15)
        if r1.returncode != 0:
            return f'compile_fail: {r1.stderr[:150]}'

        r1b = subprocess.run(["clang", "-S", "-emit-llvm", "-g", "-O0", func_c_path, "-o", func_ll_path],
                              capture_output=True, text=True, timeout=15)
        if r1b.returncode != 0:
            return f'll_gen_fail: {r1b.stderr[:150]}'

        c_deps_full = row['real_deps'] + '\n' + f"\nextern {row['func_head_types'].replace('extern','')};\n"
        c_deps_path = os.path.join(OUTDIR, f"{idx}_{fname}_deps.c")
        with open(c_deps_path, "w") as f:
            f.write(c_deps_full)
        cpp_wrapper_fixed = re.sub(
            r'extern\s"C"\s\{\s.*\s\}',
            'extern "C" \n{\n#include "' + c_deps_path + '"\n}\n',
            row['real_exe_wrapper'])
        cpp_path = os.path.join(OUTDIR, f"{idx}_{fname}_wrapper.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_wrapper_fixed)

        r2 = subprocess.run(["clang++", "-fpermissive", "-g", "-O0", "-o", exe_path,
                              cpp_path, func_obj_path, "-I", EXEBENCH_PKG_DIR],
                             capture_output=True, text=True, timeout=15)
        if r2.returncode != 0:
            return f'link_fail: {r2.stderr[:150]}'

        r3 = subprocess.run(["python3", MAPPING_SCRIPT, exe_path, func_ll_path, mapping_path],
                             capture_output=True, text=True, timeout=30)
        if r3.returncode != 0:
            return f'mapping_fail: {r3.stderr[:150]}'
        if not os.path.exists(mapping_path):
            return 'mapping_fail: no output file'

        n_inputs = len(row['real_io_pairs']['input'])
        n_traced = 0
        for k in range(n_inputs):
            try:
                inp = exebench_dict_to_dict(row['real_io_pairs']['input'][k])
                input_json = os.path.join(OUTDIR, f"{idx}_{fname}_run{k}_input.json")
                output_json = os.path.join(OUTDIR, f"{idx}_{fname}_run{k}_output.json")
                with open(input_json, 'w') as f:
                    json.dump(inp, f)

                trace_out = os.path.join(OUTDIR, "branch_trace.out")
                if os.path.exists(trace_out):
                    os.remove(trace_out)
                cmd = f'{PIN_ROOT}/pin -t {PIN_TOOL} -- {exe_path} {input_json} {output_json}'
                subprocess.run(cmd.split(), capture_output=True, text=True, timeout=15, cwd=OUTDIR)

                final_trace = os.path.join(OUTDIR, f"{idx}_{fname}_run{k}_trace.out")
                if os.path.exists(trace_out):
                    os.rename(trace_out, final_trace)
                    n_traced += 1
            except Exception:
                continue

        return f'success: {n_traced}/{n_inputs} runs traced'

    except Exception as e:
        return f'error: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    dataset = load_dataset('jordiae/exebench', split='train_real_simple_io', trust_remote_code=True)

    results = []
    for idx in range(args.start, args.start + args.count):
        row = dataset[idx]
        status = process_sample(idx, row)
        results.append((idx, row['fname'], status))
        print(f"{idx}\t{row['fname']}\t{status}")

    print("\n=== SUMMARY ===")
    success = sum(1 for _, _, s in results if s.startswith('success'))
    already = sum(1 for _, _, s in results if s == 'already_done')
    skipped = sum(1 for _, _, s in results if s == 'skip_trivial')
    failed = len(results) - success - already - skipped
    print(f"Success: {success}, Already done: {already}, Skipped (trivial): {skipped}, Failed: {failed}")
    print(f"Total attempted: {len(results)}")


if __name__ == "__main__":
    main()
