#!/usr/bin/env python3
"""Train all 8 single-task baselines sequentially on GPU"""
import subprocess, sys, os, time, datetime

TASKS = ['click', 'like', 'follow', 'comment', 'forward', 'longview', 'hate', 'profile']
TASK_NAMES = {
    'click': 'is_click', 'like': 'is_like', 'follow': 'is_follow',
    'comment': 'is_comment', 'forward': 'is_forward', 'longview': 'long_view',
    'hate': 'is_hate', 'profile': 'is_profile_enter'
}
GPU = '0'
CONFIG = './config'

results = []
total_start = time.time()

for i, task in enumerate(TASKS):
    expid = f'SingleTask_{task}'
    print(f'
{"="*60}')
    print(f'[{i+1}/8] Training: {expid} ({TASK_NAMES[task]})')
    print(f'{"="*60}')
    
    start = time.time()
    cmd = f'/root/miniconda3/bin/python run_expid.py --expid {expid} --config {CONFIG} --gpu {GPU}'
    
    ret = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start
    
    status = 'OK' if ret.returncode == 0 else f'FAIL({ret.returncode})'
    print(f'[{i+1}/8] {expid}: {status} ({elapsed:.0f}s)')
    results.append({'task': task, 'expid': expid, 'status': status, 'time': elapsed})

total = time.time() - total_start
print(f'
{"="*60}')
print(f'All done! Total: {total:.0f}s ({total/60:.1f}min)')
print(f'{"="*60}')
for r in results:
    print(f'  {r["expid"]}: {r["status"]} ({r["time"]:.0f}s)')
