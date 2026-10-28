#!/bin/bash
echo "=== monitor calculation in FuxiCTR ==="
grep -n "monitor\|val_auc\|avg_auc\|save_best\|best_metric" /root/mtl4rec/fuxictr/fuxictr/pytorch/torch_utils.py 2>/dev/null | head -20
echo ""
echo "=== run_expid.py monitor ==="
grep -n "monitor\|val_auc\|avg_auc" /root/mtl4rec/run_expid.py 2>/dev/null | head -20
echo ""
echo "=== train step logging ==="
grep -n "train.*loss\|epoch.*loss\|logging.*loss\|log.*loss" /root/mtl4rec/fuxictr/fuxictr/pytorch/torch_utils.py 2>/dev/null | head -10
grep -n "train.*loss\|epoch.*loss\|verbose\|print.*loss" /root/mtl4rec/run_expid.py 2>/dev/null | head -10