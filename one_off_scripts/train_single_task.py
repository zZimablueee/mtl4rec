#!/usr/bin/env python3
"""Train all 8 single-task models and extract AUC"""
import sys, os, subprocess, datetime, json, re

TASKS = ['click', 'like', 'follow', 'comment', 'forward', 'longview', 'hate', 'profile']
CONFIG_DIR = './config'
GPU = '0'

results = {}
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = './results/logs'
os.makedirs(log_dir, exist_ok=True)

for task in TASKS:
    expid = f'SingleTask_{task}'
    log_file = f'{log_dir}/{expid}_{timestamp}.log'
    cmd = f'/root/miniconda3/bin/python run_expid.py --expid {expid} --config {CONFIG_DIR} --gpu {GPU}'
    
    print(f'\n{"="*60}')
    print(f'Training: {expid}')
    print(f'Command: {cmd}')
    print(f'{"="*60}')
    
    tqdm_pattern = re.compile(r'^\s*\d+%\|')
    
    with open(log_file, 'w') as f:
        f.write(f'Command: {cmd}\n')
        f.write(f'Started: {datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}\n')
        f.write(f'{"="*60}\n')
        f.flush()
        
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        test_auc = None
        
        for line in proc.stdout:
            is_tqdm = bool(tqdm_pattern.match(line))
            if is_tqdm:
                stripped = re.sub(r'^\s*\d+%\|.*?\]', '', line).strip()
                if stripped:
                    print(stripped)
                    f.write(stripped + '\n')
                    f.flush()
                else:
                    print(line, end='')
            else:
                print(line, end='')
                f.write(line)
                f.flush()
                # Extract test AUC
                if 'test_result' in line.lower() or 'test AUC' in line:
                    pass
        
        proc.wait()
        f.write(f'\n{"="*60}\n')
        f.write(f'Exit code: {proc.returncode}\n')
    
    print(f'\nCompleted: {expid} (exit code: {proc.returncode})')
    results[expid] = {'exit_code': proc.returncode}

# Save results summary
summary_file = f'./results/single_task_summary_{timestamp}.json'
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSummary saved to: {summary_file}')
