#!/bin/bash
grep -rn "class Monitor" /root/mtl4rec/fuxictr/pytorch/ --include="*.py"
echo "===="
# Find monitor.py or metrics.py
find /root/mtl4rec/fuxictr/pytorch -name "*.py" | xargs grep -l "class Monitor" 2>/dev/null