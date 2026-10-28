#!/usr/bin/env python3
"""Train all 8 single-task models sequentially, save results"""
import subprocess, sys, os, json, re, datetime

TASKS = ['click', 'like', 'follow', 'comment', 'forward', 'longview', 'hate', 'profile']
CONFIG = './config'
GPU = '-1'  # CPU only

results = {}
log_dir = './results/logs'
os.makedirs(log_dir, exist_ok=True)

for i, task in enumerate(TASKS):
    expid = 'SingleTask_' + task
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir + '/' + expid + '_' + ts + '.log'
    cmd = '/root/miniconda3/bin/python run_expid.py --expid ' + expid + ' --config ' + CONFIG + ' --gpu ' + GPU
    
    print()
    print('=' * 60)
    print('[' + str(i+1) + '/8] Training: ' + expid)
    print('Command: ' + cmd)
    print('Log: ' + log_file)
    print('=' * 60)
    sys.stdout.flush()
    
    tqdm_pattern = re.compile(r'^\s*\d+%\|')
    
    with open(log_file, 'w') as f:
        f.write('Command: ' + cmd + '\n')
        f.write('Started: ' + ts + '\n')
        f.write('=' * 60 + '\n')
        f.flush()
        
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in proc.stdout:
            is_tqdm = bool(tqdm_pattern.match(line))
            if is_tqdm:
                stripped = re.sub(r'^\s*\d+%\|.*?\]', '', line).strip()
                if stripped:
                    print(stripped)
                    f.write(stripped + '\n')
                else:
                    print(line, end='')
            else:
                print(line, end='')
                f.write(line)
            f.flush()
        
        proc.wait()
        f.write('\n' + '=' * 60 + '\n')
        f.write('Exit code: ' + str(proc.returncode) + '\n')
    
    results[expid] = {
        'exit_code': proc.returncode,
        'log_file': log_file,
        'timestamp': ts
    }
    print()
    print('[' + str(i+1) + '/8] Completed: ' + expid + ' (exit: ' + str(proc.returncode) + ')')

# Save summary
summary_file = './results/single_task_summary_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)
print()
print('All done! Summary: ' + summary_file)