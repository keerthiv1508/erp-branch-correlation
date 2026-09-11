#!/usr/bin/env python3
"""
Parallel version of run_pin_batch.py using multiprocessing.Pool. 
Each worker gets its own /tmp working directory to avoid branch_trace.out
Filename collisions between concurrent Pin processes.
"""

from datasets import load_dataset
from exebench import Wrapper, exebench_dict_to_dict
import json
import subprocess
import os
import time
import multiprocessing
import shutil

PIN_ROOT = '/home_alt/keerthivasan/software/pin-3.28-98749-g6643ecee5-gcc-linux'
PIN_TOOL = f'{PIN_ROOT}/source/tools/ManualExamples/obj-intel64/branch_trace.so'
OUTPUT_DIR = '/home_hdd/keerthivasan/exebench/traces_parallel'
NUM_WORKERS = 20
TIMEOUT_S = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_row(args):
        """Runs in a worker process. Returns 'success', 'fail', or 'skip'."""
    idx, row = args
    if len(row['func_def']) < 100:
        return 'skip'

    worker_dir = f"/tmp/pin_worker_{os.getpid()}"
    os.makedirs(worker_dir, exist_ok=True)
    trace_path = os.path.join(worker_dir, 'branch_trace.out')

    try:
        wrapper = Wrapper(
            c_deps=row['real_deps'] + '\n',
            func_c_signature=row['func_head_types'].replace('extern', ''),
            func_assembly=row['asm']['code'][0],
            cpp_wrapper=row['real_exe_wrapper']
        )
        exe_path = str(wrapper._compiled_exe_path)
        inp = exebench_dict_to_dict(row['real_io_pairs']['input'][0])

        input_json = os.path.join(worker_dir, f'{idx}_{row["fname"]}_input.json')
        output_json = os.path.join(worker_dir, f'{idx}_{row["fname"]}_output.json')
        with open(input_json, 'w') as f:
            json.dump(inp, f)

        if os.path.exists(trace_path):
            os.remove(trace_path)

        cmd = f'{PIN_ROOT}/pin -t {PIN_TOOL} -- {exe_path} {input_json} {output_json}'
        subprocess.run(cmd.split(), capture_output=True, text=True,
                        timeout=TIMEOUT_S, cwd=worker_dir)

        if os.path.exists(trace_path):
            dest = os.path.join(OUTPUT_DIR, f'{idx}_{row["fname"]}_trace.out')
            shutil.move(trace_path, dest)
            return 'success'
        else:
            return 'fail'
    except Exception:
        return 'fail'


def main():
    n_samples = None  # start with a measured subset, extrapolate from there
    split_str = 'train_real_simple_io' if n_samples is None else f'train_real_simple_io[:{n_samples}]'
    dataset = load_dataset('jordiae/exebench',
                            split=split_str,
                            trust_remote_code=True)
    rows = list(enumerate(dataset))

    start_time = time.time()
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_row, rows)
    elapsed = time.time() - start_time

    skip_count = results.count('skip')
    success_count = results.count('success')
    fail_count = results.count('fail')
    attempted = success_count + fail_count

    print(f'\n--- PARALLEL SUMMARY ({NUM_WORKERS} workers) ---')
    print(f'Total rows checked: {len(rows)}')
    print(f'Skipped (trivial): {skip_count}')
    print(f'Attempted (compile+trace): {attempted}')
    print(f'Successful: {success_count}')
    print(f'Failed: {fail_count}')
    print(f'Success rate: {round(100*success_count/attempted, 1) if attempted else 0}%')
    print(f'Total wall time: {round(elapsed, 1)}s')
    print(f'Effective time per attempted sample: {round(elapsed/attempted, 3) if attempted else 0}s')


if __name__ == '__main__':
    main()
