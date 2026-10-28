#!/bin/bash
grep -rn "monitor" /root/mtl4rec/fuxictr/pytorch/ --include="*.py" | head -20
echo "===="
grep -rn "val_auc\|avg_auc\|save_best" /root/mtl4rec/fuxictr/pytorch/ --include="*.py" | head -20
echo "===="
grep -rn "train_loss\|epoch_loss\|loss.*log\|log.*loss" /root/mtl4rec/fuxictr/pytorch/ --include="*.py" | head -10
echo "===="
grep -rn "monitor\|val_auc\|save_best" /root/mtl4rec/run_expid.py | head -10