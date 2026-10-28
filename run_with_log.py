#!/usr/bin/env python3
"""Wrapper: run training, save clean log (keep Train loss, strip tqdm bars)"""
import sys, os, subprocess, datetime, re

expid = sys.argv[1]
config_dir = sys.argv[2] if len(sys.argv) > 2 else "./config"
gpu = sys.argv[3] if len(sys.argv) > 3 else "0"
config_version = sys.argv[4] if len(sys.argv) > 4 else "v3"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "./results/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = f"{log_dir}/{expid}_{config_version}_{timestamp}.log"

cmd = f"/root/miniconda3/bin/python run_expid.py --expid {expid} --config {config_dir} --gpu {gpu}"
print(f"Running: {cmd}")
print(f"Config version: {config_version}")
print(f"Log saved to: {log_file}")

# tqdm progress bar pattern: " 93%|..." or "100%|..."
tqdm_pattern = re.compile(r'^\s*\d+%\|')

with open(log_file, 'w') as f:
    f.write(f"Config version: {config_version}\n")
    f.write(f"Command: {cmd}\n")
    f.write(f"Started: {timestamp}\n")
    f.write(f"{'='*60}\n")
    f.flush()
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        is_tqdm = bool(tqdm_pattern.match(line))
        if is_tqdm:
            # Strip tqdm prefix, keep useful info after it (e.g. Train loss)
            # tqdm line format: "  N%|...| N/N [time]INFO Train loss: X.XXX"
            stripped = re.sub(r'^\s*\d+%\|.*?\]', '', line).strip()
            if stripped:
                # Has useful info after tqdm bar (e.g., Train loss)
                print(stripped)
                f.write(stripped + '\n')
                f.flush()
            else:
                # Pure tqdm bar, only print to terminal
                print(line, end='')
        else:
            print(line, end='')
            f.write(line)
            f.flush()
    proc.wait()
    f.write(f"\n{'='*60}\n")
    f.write(f"Exit code: {proc.returncode}\n")
    f.write(f"Config version: {config_version}\n")

print(f"\nLog saved to: {log_file}")