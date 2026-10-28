#!/bin/bash
# ============================================================
# 一键搭建 KuaiRand-1K 多任务对比实验环境
# 在服务器上运行: bash setup.sh
# ============================================================
set -e

PROJECT_DIR="/root/mtl4rec"
STEM_REPO="https://github.com/LiangcaiSu/STEM.git"
DATA_DIR="/root/autodl-fs/KuaiRand-1K/data"

echo "=============================================="
echo " KuaiRand-1K Multi-Task Setup"
echo "=============================================="

# 1. 进入项目目录
echo "[1/6] 进入项目目录 $PROJECT_DIR"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# 2. 安装依赖
echo "[2/6] 安装 Python 依赖"
pip install fuxictr pandas numpy scikit-learn h5py pyyaml tqdm -q

# 3. 克隆 STEM 仓库（临时，只取模型代码）
echo "[3/6] 克隆 STEM 仓库获取模型代码"
TMPDIR=$(mktemp -d)
git clone --depth 1 $STEM_REPO $TMPDIR/STEM

# 复制 models/ 目录（SharedBottom, MMoE, PLE 等）
echo "      复制 models/ 到 $PROJECT_DIR/models/"
cp -r $TMPDIR/STEM/models $PROJECT_DIR/models

# 复制 run_expid.py 训练入口
cp $TMPDIR/STEM/run_expid.py $PROJECT_DIR/run_expid.py

# 清理临时仓库
rm -rf $TMPDIR
echo "      模型代码已就绪"

# 4. 创建 config 目录和配置文件
echo "[4/6] 创建 config 配置文件"
mkdir -p $PROJECT_DIR/config

# 写入 dataset_config.yaml
cat > $PROJECT_DIR/config/dataset_config.yaml << 'YAML_EOF'
# KuaiRand-1K 数据集配置
KuaiRand1K:
  data_root: ./data
  data_format: csv
  train_data: ./data/KuaiRand1K/train.csv
  valid_data: ./data/KuaiRand1K/valid.csv
  test_data: ./data/KuaiRand1K/test.csv
  min_categr_count: 1
  feature_cols:
    - name: [user_id, video_id, author_id, bgm_id, tab, video_type,
             upload_type, visible, user_active_degree,
             is_live_streamer, is_video_author]
      active: True
      dtype: str
      type: categorical
      fill_na: ""
    - name: [follow, fan, friend, active_day, duration_ms]
      active: True
      dtype: float
      type: numeric
      normalizer: standard
  label_col:
    - name: is_click
      dtype: float
    - name: is_like
      dtype: float
  group_id: 'user_id'
YAML_EOF

# 写入 model_config.yaml
cat > $PROJECT_DIR/config/model_config.yaml << 'YAML_EOF'
# =============================================
# 多任务对比实验 (2任务: click + like)
# =============================================

SharedBottom_KuaiRand1K:
    model: SharedBottom
    dataset_id: KuaiRand1K
    task_labels: ['is_click', 'is_like']
    loss: ['binary_crossentropy', 'binary_crossentropy']
    metrics: ['logloss', 'AUC', 'gAUC']
    task: ['binary_classification', 'binary_classification']
    num_tasks: 2
    optimizer: adam
    learning_rate: 1.e-4
    bottom_hidden_units: [512, 512, 512]
    tower_hidden_units: [128, 128]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-6
    batch_norm: False
    net_dropout: 0
    batch_size: 4096
    embedding_dim: 16
    epochs: 20
    shuffle: True
    seed: 2023
    monitor: 'AUC'
    monitor_mode: 'max'
    group_id: 'user_id'
    model_root: './checkpoints'
    num_workers: 3
    verbose: 1
    early_stop_patience: 2
    save_best_only: True

MMoE_KuaiRand1K:
    model: MMoE
    dataset_id: KuaiRand1K
    task_labels: ['is_click', 'is_like']
    loss: ['binary_crossentropy', 'binary_crossentropy']
    metrics: ['logloss', 'AUC', 'gAUC']
    task: ['binary_classification', 'binary_classification']
    num_tasks: 2
    optimizer: adam
    learning_rate: 1.e-4
    num_experts: 4
    expert_hidden_units: [512, 256, 128]
    gate_hidden_units: [128, 64]
    tower_hidden_units: [128, 64]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-6
    batch_norm: False
    net_dropout: 0
    batch_size: 4096
    embedding_dim: 16
    epochs: 20
    shuffle: True
    seed: 2023
    monitor: 'AUC'
    monitor_mode: 'max'
    group_id: 'user_id'
    model_root: './checkpoints'
    num_workers: 3
    verbose: 1
    early_stop_patience: 2
    save_best_only: True

PLE_KuaiRand1K:
    model: PLE
    dataset_id: KuaiRand1K
    task_labels: ['is_click', 'is_like']
    loss: ['binary_crossentropy', 'binary_crossentropy']
    metrics: ['logloss', 'AUC', 'gAUC']
    task: ['binary_classification', 'binary_classification']
    num_tasks: 2
    optimizer: adam
    learning_rate: 1.e-4
    num_layers: 1
    num_shared_experts: 4
    num_specific_experts: 1
    expert_hidden_units: [512, 256, 128]
    gate_hidden_units: [128, 64]
    tower_hidden_units: [128, 64]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-6
    batch_norm: False
    net_dropout: 0
    batch_size: 4096
    embedding_dim: 16
    epochs: 20
    shuffle: True
    seed: 2023
    monitor: 'AUC'
    monitor_mode: 'max'
    group_id: 'user_id'
    model_root: './checkpoints'
    num_workers: 3
    verbose: 1
    early_stop_patience: 2
    save_best_only: True

# 单任务基线
SharedBottom_KuaiRand1K_click:
    model: SharedBottom
    dataset_id: KuaiRand1K
    task_labels: ['is_click']
    loss: ['binary_crossentropy']
    metrics: ['logloss', 'AUC', 'gAUC']
    task: ['binary_classification']
    num_tasks: 1
    optimizer: adam
    learning_rate: 1.e-4
    bottom_hidden_units: [512, 512, 512]
    tower_hidden_units: [128, 128]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-6
    batch_norm: False
    net_dropout: 0
    batch_size: 4096
    embedding_dim: 16
    epochs: 20
    shuffle: True
    seed: 2023
    monitor: 'AUC'
    monitor_mode: 'max'
    group_id: 'user_id'
    model_root: './checkpoints'
    num_workers: 3
    verbose: 1
    early_stop_patience: 2
    save_best_only: True

SharedBottom_KuaiRand1K_like:
    model: SharedBottom
    dataset_id: KuaiRand1K
    task_labels: ['is_like']
    loss: ['binary_crossentropy']
    metrics: ['logloss', 'AUC', 'gAUC']
    task: ['binary_classification']
    num_tasks: 1
    optimizer: adam
    learning_rate: 1.e-4
    bottom_hidden_units: [512, 512, 512]
    tower_hidden_units: [128, 128]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-6
    batch_norm: False
    net_dropout: 0
    batch_size: 4096
    embedding_dim: 16
    epochs: 20
    shuffle: True
    seed: 2023
    monitor: 'AUC'
    monitor_mode: 'max'
    group_id: 'user_id'
    model_root: './checkpoints'
    num_workers: 3
    verbose: 1
    early_stop_patience: 2
    save_best_only: True
YAML_EOF

echo "      配置文件已创建: config/dataset_config.yaml, config/model_config.yaml"

# 5. 创建数据预处理脚本
echo "[5/6] 创建数据预处理脚本"

cat > $PROJECT_DIR/kuairand_preprocess.py << 'PYEOF'
#!/usr/bin/env python
"""KuaiRand-1K 数据预处理"""
import pandas as pd
import numpy as np
import os, argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/root/autodl-fs/KuaiRand-1K/data')
    p.add_argument('--output_dir', default='./data/KuaiRand1K')
    p.add_argument('--valid_ratio', type=float, default=0.1)
    p.add_argument('--test_ratio', type=float, default=0.1)
    return p.parse_args()

def load_user_features(path):
    print(f"Loading user features: {path}")
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype in [np.int64, np.float64]:
            df[col] = df[col].replace(-124, np.nan)
    df['is_live_streamer'] = df['is_live_streamer'].fillna(0).astype(int)
    df['is_video_author'] = df['is_video_author'].fillna(0).astype(int)
    df['user_active_degree'] = df['user_active_degree'].astype(str)
    return df

def load_video_features(basic_path, stat_path=None):
    print(f"Loading video features: {basic_path}")
    df = pd.read_csv(basic_path)
    if stat_path and os.path.exists(stat_path):
        df_stat = pd.read_csv(stat_path)
        df = df.merge(df_stat, on='video_id', how='left')
    return df

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load user and video features
    df_user = load_user_features(os.path.join(args.data_dir, 'user_features_1k.csv'))
    df_video = load_video_features(
        os.path.join(args.data_dir, 'video_features_basic_1k.csv'),
        os.path.join(args.data_dir, 'video_features_statistic_1k.csv'))

    # Load interaction log
    log_path = os.path.join(args.data_dir, 'log_standard_4_22_to_5_08_1k.csv')
    print(f"Loading interaction log: {log_path}")
    df_log = pd.read_csv(log_path)
    print(f"Total interactions: {len(df_log):,}")

    # Merge user and video features
    df = df_log.merge(df_user, on='user_id', how='left')
    df = df.merge(df_video, on='video_id', how='left')
    print(f"After merging: {len(df):,} rows, {len(df.columns)} cols")

    # Select features and labels
    feature_cols = [
        'user_id', 'video_id', 'author_id', 'bgm_id', 'tab',
        'video_type', 'upload_type', 'visible',
        'follow', 'fan', 'friend', 'active_day', 'user_active_degree',
        'is_live_streamer', 'is_video_author', 'duration_ms',
    ]
    label_cols = ['is_click', 'is_like']

    # Check which stat features exist
    for col in df.columns:
        if col not in feature_cols and col not in label_cols:
            feature_cols.append(col)
            if len(feature_cols) > 40:
                break

    keep_cols = [c for c in feature_cols if c in df.columns] + label_cols
    df_final = df[keep_cols].copy()
    print(f"Final dataset: {len(df_final):,} rows, {len(keep_cols)} cols")
    print(f"Labels: {label_cols}")

    # Split by time order
    total = len(df_final)
    test_n = int(total * args.test_ratio)
    valid_n = int(total * args.valid_ratio)
    train_n = total - valid_n - test_n

    train = df_final.iloc[:train_n].reset_index(drop=True)
    valid = df_final.iloc[train_n:train_n+valid_n].reset_index(drop=True)
    test = df_final.iloc[train_n+valid_n:].reset_index(drop=True)

    print(f"Train: {len(train):,} | Valid: {len(valid):,} | Test: {len(test):,}")

    train.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    valid.to_csv(os.path.join(args.output_dir, 'valid.csv'), index=False)
    test.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    print(f"Saved to {args.output_dir}/")
    print("Done!")

if __name__ == '__main__':
    main()
PYEOF

echo "      预处理脚本已创建: kuairand_preprocess.py"

# 6. 创建 results/ 目录
mkdir -p $PROJECT_DIR/results

echo ""
echo "=============================================="
echo " 全部就绪！"
echo "=============================================="
echo ""
echo " 项目目录: $PROJECT_DIR"
echo ""
echo " 目录结构:"
echo "   $PROJECT_DIR/"
echo "   ├── config/"
echo "   │   ├── dataset_config.yaml"
echo "   │   └── model_config.yaml"
echo "   ├── models/          (从 STEM 复制的模型代码)"
echo "   ├── run_expid.py     (训练入口)"
echo "   ├── kuairand_preprocess.py"
echo "   ├── data/            (预处理后生成)"
echo "   ├── checkpoints/     (训练后生成)"
echo "   └── results/         (评估结果)"
echo ""
echo " 接下来运行:"
echo ""
echo "  # 第1步: 数据预处理"
echo "  python kuairand_preprocess.py \\"
echo "      --data_dir $DATA_DIR \\"
echo "      --output_dir ./data/KuaiRand1K"
echo ""
echo "  # 第2步: 训练"
echo "  python run_expid.py --expid SharedBottom_KuaiRand1K --config ./config --gpu 0"
echo "  python run_expid.py --expid MMoE_KuaiRand1K --config ./config --gpu 0"
echo "  python run_expid.py --expid PLE_KuaiRand1K --config ./config --gpu 0"