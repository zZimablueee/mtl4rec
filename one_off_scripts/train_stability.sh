#!/bin/bash
# ============================================================
# 稳定性分析训练脚本
# 先跑 V5 (静态权重), 再跑 V7 (不确定性加权)
# ============================================================
set -e

PROJECT_DIR="/root/mtl4rec"
CONFIG_DIR="$PROJECT_DIR/config"
RESULTS_DIR="$PROJECT_DIR/results/stability_analysis"
ORIG_CONFIG="$CONFIG_DIR/model_config.yaml"
BACKUP_CONFIG="$CONFIG_DIR/model_config.yaml.main_backup"

cd $PROJECT_DIR

# 备份原始 config
if [ ! -f "$BACKUP_CONFIG" ]; then
    cp "$ORIG_CONFIG" "$BACKUP_CONFIG"
    echo "[$(date)] Backed up original model_config.yaml"
fi

# ============================================================
# V5: 静态 loss_weight
# ============================================================
echo "========================================="
echo "[$(date)] Starting V5 training (static loss_weight)"
echo "========================================="

cp "$CONFIG_DIR/model_config_stability_v5.yaml" "$ORIG_CONFIG"
mkdir -p "$RESULTS_DIR/v5_static"

# 清理旧 checkpoint (避免冲突)
rm -f "$PROJECT_DIR/checkpoints/KuaiRand1K/PLE_stability_KuaiRand1K.model"
rm -f "$PROJECT_DIR/checkpoints/KuaiRand1K/PLE_stability_KuaiRand1K.log"

/root/miniconda3/bin/python run_with_log.py \
    PLE_stability_KuaiRand1K \
    ./config \
    0 \
    stability_v5

echo "[$(date)] V5 training completed"

# ============================================================
# V7: 不确定性加权
# ============================================================
echo "========================================="
echo "[$(date)] Starting V7 training (uncertainty weighting)"
echo "========================================="

cp "$CONFIG_DIR/model_config_stability_v7.yaml" "$ORIG_CONFIG"
mkdir -p "$RESULTS_DIR/v7_uncertainty"

# 清理旧 checkpoint
rm -f "$PROJECT_DIR/checkpoints/KuaiRand1K/PLE_v7_stability_KuaiRand1K.model"
rm -f "$PROJECT_DIR/checkpoints/KuaiRand1K/PLE_v7_stability_KuaiRand1K.log"

/root/miniconda3/bin/python run_with_log.py \
    PLE_v7_stability_KuaiRand1K \
    ./config \
    0 \
    stability_v7

echo "[$(date)] V7 training completed"

# ============================================================
# 恢复原始 config
# ============================================================
cp "$BACKUP_CONFIG" "$ORIG_CONFIG"
echo "[$(date)] Restored original model_config.yaml"
echo "========================================="
echo "Training complete! Results in $RESULTS_DIR"
echo "========================================="