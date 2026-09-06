#!/usr/bin/env python3
import glob
import json
import os
import subprocess

from datasets import load_dataset
from exebench import exebench_dict_to_dict

OUTDIR = os.path.expanduser("~/clang_pipeline_test")
PIN_ROOT = '/home_alt/keerthivasan/software/pin-3.28-98749-g6643ecee5-gcc-linux'
PIN_TOOL = f'{PIN_ROOT}/source/tools/ManualExamples/obj-intel64/branch_trace.so'

exe_files = sorted(glob.glob(os.path.join(OUTDIR, "*_clang_exe")))
print(f"Found {len(exe_files)} executables")

dataset = load_dataset('jordiae/exebench', split='train_real_simple_io', trust_remote_code=True)

results = []
for exe_path in exe_files:
    base = os.path.basename(exe_path).replace("_clang_exe", "")
    idx, fname = base.split("_", 1)
    row = dataset[int(idx)]
    n_inputs = len(row['real_io_pairs']['input'])

    print(f"\n=== Sample {idx} ({fname}): {n_inputs} inputs available ===")
    n_success = 0

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
            r = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=15, cwd=OUTDIR)

            final_trace = os.path.join(OUTDIR, f"{idx}_{fname}_run{k}_trace.out")
            if os.path.exists(trace_out):
                os.rename(trace_out, final_trace)
                n_success += 1
            else:
                print(f"    run {k}: FAILED (no trace produced, stderr: {r.stderr[:100]})")
        except Exception as e:
            print(f"    run {k}: ERROR {e}")

    print(f"  {n_success}/{n_inputs} runs succeeded")
    results.append((idx, fname, n_inputs, n_success))

print("\n=== SUMMARY ===")
total_runs = sum(r[3] for r in results)
total_attempted = sum(r[2] for r in results)
for idx, fname, n_inputs, n_success in results:
    print(f"{idx}\t{fname}\t{n_success}/{n_inputs}")
print(f"\nTotal: {total_runs}/{total_attempted} runs succeeded across {len(results)} samples")
