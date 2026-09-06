#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys

_EXEBENCH_PKG_DIR = "/home_hdd/keerthivasan/exebench/exebench"


def run_cmd(cmd, label):
    print(f"--- {label} ---")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print("stdout:", result.stdout.strip())
    if result.stderr.strip():
        print("stderr:", result.stderr.strip())
    if result.returncode != 0:
        print(f"FAILED (exit code {result.returncode})")
        sys.exit(1)
    print("OK")
    print()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_index", type=int)
    parser.add_argument("--outdir", default=os.path.expanduser("~/clang_pipeline_test"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from datasets import load_dataset
    dataset = load_dataset('jordiae/exebench', split='train_real_simple_io', trust_remote_code=True)
    row = dataset[args.sample_index]
    fname = row['fname']
    print(f"Sample {args.sample_index}: {fname}")
    print()

    func_def = row['func_def']
    func_c_signature = row['func_head_types'].replace('extern', '')
    c_deps = row['real_deps'] + '\n'
    cpp_wrapper = row['real_exe_wrapper']

    func_c_path = os.path.join(args.outdir, f"{args.sample_index}_{fname}_func.c")
    func_obj_path = os.path.join(args.outdir, f"{args.sample_index}_{fname}_func.o")
    with open(func_c_path, "w") as f:
        f.write(func_def)

    run_cmd(["clang", "-g", "-O0", "-c", func_c_path, "-o", func_obj_path],
            "Step 1: compile function source")

    c_deps_full = c_deps + f'\nextern {func_c_signature};\n'
    c_deps_path = os.path.join(args.outdir, f"{args.sample_index}_{fname}_deps.c")
    with open(c_deps_path, "w") as f:
        f.write(c_deps_full)

    cpp_wrapper_fixed = re.sub(
        r'extern\s"C"\s\{\s.*\s\}',
        'extern "C" \n{\n#include "' + c_deps_path + '"\n}\n',
        cpp_wrapper
    )
    cpp_path = os.path.join(args.outdir, f"{args.sample_index}_{fname}_wrapper.cpp")
    with open(cpp_path, "w") as f:
        f.write(cpp_wrapper_fixed)

    executable_path = os.path.join(args.outdir, f"{args.sample_index}_{fname}_clang_exe")
    run_cmd(
        ["clang++", "-fpermissive", "-g", "-O0", "-o", executable_path,
         cpp_path, func_obj_path,
         "-I", _EXEBENCH_PKG_DIR, "-I", _EXEBENCH_PKG_DIR],
        "Step 2: compile wrapper + link"
    )

    print(f"SUCCESS. Executable: {executable_path}")
    print(f"Function object: {func_obj_path}")


if __name__ == "__main__":
    main()
