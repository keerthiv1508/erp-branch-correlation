#!/usr/bin/env python3
import glob, os, re, subprocess, sys, json
from datasets import load_dataset
from exebench import exebench_dict_to_dict

MAPPING_DIR = "/home_alt/keerthivasan/mapping_json"
OUTDIR = os.path.expanduser("~/clang_pipeline_test")
PIN_ROOT = '/home_alt/keerthivasan/software/pin-3.28-98749-g6643ecee5-gcc-linux'
PIN_TOOL = f'{PIN_ROOT}/source/tools/ManualExamples/obj-intel64/branch_trace.so'
EXEBENCH_PKG_DIR = "/home_hdd/keerthivasan/exebench/exebench"
os.makedirs(OUTDIR, exist_ok=True)

indices = sorted(int(re.search(r'sample_(\d+)\.json', f).group(1))
                  for f in glob.glob(os.path.join(MAPPING_DIR, "sample_*.json")))
print(f"Found {len(indices)} samples with mapping JSONs: {indices}")

dataset = load_dataset('jordiae/exebench', split='train_real_simple_io', trust_remote_code=True)

results = []
for idx in indices:
    row = dataset[idx]
    fname = row['fname']
    print(f"\n=== Sample {idx}: {fname} ===")
    try:
        func_c_path = os.path.join(OUTDIR, f"{idx}_{fname}_func.c")
        func_obj_path = os.path.join(OUTDIR, f"{idx}_{fname}_func.o")
        with open(func_c_path, "w") as f:
            f.write(row['func_def'])
        r1 = subprocess.run(["clang", "-g", "-O0", "-c", func_c_path, "-o", func_obj_path],
                             capture_output=True, text=True)
        if r1.returncode != 0:
            print(f"  compile FAILED: {r1.stderr[:200]}")
            results.append((idx, fname, 'compile_fail'))
            continue

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

        exe_path = os.path.join(OUTDIR, f"{idx}_{fname}_clang_exe")
        r2 = subprocess.run(["clang++", "-fpermissive", "-g", "-O0", "-o", exe_path,
                              cpp_path, func_obj_path,
                              "-I", EXEBENCH_PKG_DIR],
                             capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"  link FAILED: {r2.stderr[:300]}")
            results.append((idx, fname, 'link_fail'))
            continue

        inp = exebench_dict_to_dict(row['real_io_pairs']['input'][0])
        input_json = os.path.join(OUTDIR, f"{idx}_{fname}_input.json")
        output_json = os.path.join(OUTDIR, f"{idx}_{fname}_output.json")
        with open(input_json, 'w') as f:
            json.dump(inp, f)

        trace_out = os.path.join(OUTDIR, "branch_trace.out")
        if os.path.exists(trace_out):
            os.remove(trace_out)
        cmd = f'{PIN_ROOT}/pin -t {PIN_TOOL} -- {exe_path} {input_json} {output_json}'
        r3 = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30, cwd=OUTDIR)
        final_trace = os.path.join(OUTDIR, f"{idx}_{fname}_clang_trace.out")
        if os.path.exists(trace_out):
            os.rename(trace_out, final_trace)
            print(f"  SUCCESS -> {final_trace}")
            results.append((idx, fname, 'success'))
        else:
            print(f"  trace FAILED (no output file, stderr: {r3.stderr[:150]})")
            results.append((idx, fname, 'trace_fail'))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append((idx, fname, 'error'))

print("\n=== SUMMARY ===")
for idx, fname, status in results:
    print(f"{idx}\t{fname}\t{status}")
success = sum(1 for _,_,s in results if s == 'success')
print(f"\n{success}/{len(results)} succeeded")
