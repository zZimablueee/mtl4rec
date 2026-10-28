#!/bin/bash
echo "=== V5 status ==="
tail -1 /root/mtl4rec/results/config.csv
echo ""
echo "=== V5 latest AUCs ==="
grep "AUC" /root/mtl4rec/results/logs/v5_screen.log | tail -20
echo ""
echo "=== V5 stop info ==="
grep -i "stop\|early\|best" /root/mtl4rec/results/logs/v5_screen.log | tail -10
echo ""
echo "=== V6/V7 log files ==="
ls -la /root/mtl4rec/results/logs/PLE_KuaiRand1K_v6* /root/mtl4rec/results/logs/PLE_KuaiRand1K_v7* 2>/dev/null
echo ""
echo "=== Screen sessions ==="
screen -ls | grep -v Dead
echo ""
echo "=== Active processes ==="
ps aux | grep run_expid | grep -v grep | wc -l