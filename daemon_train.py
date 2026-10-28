import os, sys, subprocess, time

os.chdir('/root/mtl4rec')
timestamp = time.strftime('%Y%m%d_%H%M%S')
logfile = f'results/logs/PLE_v14_KuaiRand1K_v14_{timestamp}.log'

# Double fork to daemonize
if os.fork() > 0:
    sys.exit(0)
os.setsid()
if os.fork() > 0:
    sys.exit(0)

with open(logfile, 'w') as f:
    f.write(f'Config version: v14\n')
    f.write(f'Command: /root/miniconda3/bin/python run_expid.py --expid PLE_v14_KuaiRand1K --config ./config --gpu 0\n')
    f.write(f'Started: {timestamp}\n')
    f.write('='*60 + '\n')
    f.flush()
    proc = subprocess.Popen(
        ['/root/miniconda3/bin/python', 'run_expid.py', '--expid', 'PLE_v14_KuaiRand1K', '--config', './config', '--gpu', '0'],
        stdout=f, stderr=subprocess.STDOUT
    )
    proc.wait()
    f.write('='*60 + '\n')
    f.write(f'Exit code: {proc.returncode}\n')
    f.write(f'Config version: v14\n')
