#!/bin/bash
echo "=== searching monitor ==="
grep -rn "monitor" /root/mtl4rec/fuxictr/fuxictr/pytorch/ 2>/dev/null | grep -v ".pyc" | head -20
echo ""
echo "=== searching val_auc ==="
grep -rn "val_auc\|avg_auc\|metric.*mean\|mean.*metric" /root/mtl4rec/fuxictr/fuxictr/pytorch/ 2>/dev/null | grep -v ".pyc" | head -20
echo ""
echo "=== searching loss logging in train ==="
grep -rn "train_loss\|epoch_loss\|log_loss\|loss.*epoch" /root/mtl4rec/fuxictr/fuxictr/pytorch/ 2>/dev/null | grep -v ".pyc" | head -10
echo ""
echo "=== FuxiCTR file list ==="
find /root/mtl4rec/fuxictr/fuxictr/pytorch -name "*.py" -not -name "*.pyc" | head -20