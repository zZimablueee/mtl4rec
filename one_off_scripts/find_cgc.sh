#!/bin/bash
# Find CGC implementation in fuxictr
find /root/mtl4rec/fuxictr -name "*.py" | xargs grep -l "class CGC" 2>/dev/null | head -5
echo "---"
grep -rn "class CGC" /root/mtl4rec/fuxictr/ 2>/dev/null | head -5
echo "---"
grep -rn "class CGC" /root/mtl4rec/models/ 2>/dev/null | head -5