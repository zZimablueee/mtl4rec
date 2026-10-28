#!/usr/bin/env python3
import subprocess, sys, os, time, datetime

TASKS = ['click', 'like', 'follow', 'comment', 'forward', 'longview', 'hate', 'profile']
GPU = '0'
CONFIG = './config'

results = []
total_start = time.time()
SEP = '=' * 60

for i, task in enumerate(TASKS):
    expid = 'SingleTask_' + task
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print()
    print(SEP)
    print('[{}/8] {} @ {}'.format(i+1, expid, ts))
    print(SEP)
    sys.stdout.flush()
    
    start = time.time()
    cmd = '/root/miniconda3/bin/python run_expid.py --expid {} --config {} --gpu {}'.format(expid, CONFIG, GPU)
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    
    status = 'OK' if ret.returncode == 0 else 'FAIL({})'.format(ret.returncode)
    print('[{}/8] {}: {} ({:.0f}s)'.format(i+1, expid, status, elapsed))
    sys.stdout.flush()
    results.append({'task': task, 'expid': expid, 'status': status, 'time': elapsed})

total = time.time() - total_start
print()
print(SEP)
print('ALL DONE! Total: {:.0f}s ({:.1f}min)'.format(total, total/60))
for r in results:
    print('  {}: {} ({:.0f}s)'.format(r['expid'], r['status'], r['time']))
