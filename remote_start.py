#!/usr/bin/env python3
"""Start training via remote Python script"""
import subprocess
import os
import sys

os.chdir('/root/mtl4rec')
log_file = f'/root/mtl4rec/results/logs/PLE_v14_KuaiRand1K_v14_rerun.log'

with open(log_file, 'w') as f:
    proc = subprocess.Popen(
        [sys.executable, 'run_with_log.py', 'PLE_v14_KuaiRand1K', './config', '0', 'v14'],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True
    )
    print(f'PID={proc.pid}', flush=True)