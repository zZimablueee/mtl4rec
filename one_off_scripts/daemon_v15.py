import os, sys, subprocess, time, signal
os.chdir('/root/mtl4rec')
if os.fork(): sys.exit(0)
os.setsid()
if os.fork(): sys.exit(0)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
ts = time.strftime('%Y%m%d_%H%M%S')
log = f'results/logs/PLE_v15_KuaiRand1K_v15_{ts}.log'
with open(log,'w') as f:
    f.write(f'Started: {ts}
')
    f.flush()
    p = subprocess.Popen(['/root/miniconda3/bin/python','run_expid.py','--expid','PLE_v15_KuaiRand1K','--config','./config','--gpu','0'], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    p.wait()
    f.write(f'
Exit code: {p.returncode}
')
