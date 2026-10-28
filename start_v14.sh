#!/bin/bash
cd /root/mtl4rec
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE=results/logs/PLE_v14_KuaiRand1K_v14_${TIMESTAMP}.log
echo "Config version: v14" > $LOGFILE
echo "Command: /root/miniconda3/bin/python run_expid.py --expid PLE_v14_KuaiRand1K --config ./config --gpu 0" >> $LOGFILE
echo "Started: ${TIMESTAMP}" >> $LOGFILE
echo "============================================================" >> $LOGFILE
/root/miniconda3/bin/python run_expid.py --expid PLE_v14_KuaiRand1K --config ./config --gpu 0 >> $LOGFILE 2>&1
echo "============================================================" >> $LOGFILE
echo "Exit code: $?" >> $LOGFILE
echo "Config version: v14" >> $LOGFILE
