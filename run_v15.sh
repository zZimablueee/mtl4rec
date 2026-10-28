#!/bin/bash
cd /root/mtl4rec
rm -rf checkpoints/PLE_v15_KuaiRand1K
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE=results/logs/PLE_v15_KuaiRand1K_v15_${TIMESTAMP}.log
echo "Started: ${TIMESTAMP}" > $LOGFILE
/root/miniconda3/bin/python run_expid.py --expid PLE_v15_KuaiRand1K --config ./config --gpu 0 >> $LOGFILE 2>&1
echo "Exit code: $?" >> $LOGFILE
