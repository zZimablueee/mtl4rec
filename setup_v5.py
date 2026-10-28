#!/usr/bin/env python3
"""
V5 控制变量实验: 以 V3 为基线，只改 PLE num_layers: 1 → 2
使用方法: /root/miniconda3/bin/python setup_v5.py
"""
import os
import shutil

BASE = '/root/mtl4rec'
CONFIG_DIR = f'{BASE}/config'
CACHE_DIR = f'{BASE}/data/KuaiRand1K'

# 1. 备份当前 V4 缓存
v4_dir = f'{CACHE_DIR}/cache_v4'
if not os.path.exists(v4_dir):
    os.makedirs(v4_dir, exist_ok=True)
    for f in ['train.h5', 'valid.h5', 'test.h5', 'feature_map.json', 'feature_processor.pkl']:
        src = f'{CACHE_DIR}/{f}'
        if os.path.exists(src):
            shutil.copy2(src, f'{v4_dir}/{f}')
    print(f'[OK] V4 cache backup -> {v4_dir}')

# 2. 恢复 V3 缓存 (embedding_dim=16 与 V5 一致)
v3_dir = f'{CACHE_DIR}/cache_v3'
if os.path.exists(v3_dir):
    for f in ['train.h5', 'valid.h5', 'test.h5', 'feature_map.json', 'feature_processor.pkl']:
        src = f'{v3_dir}/{f}'
        dst = f'{CACHE_DIR}/{f}'
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print(f'[OK] V3 cache restored -> {CACHE_DIR}')
else:
    print(f'[WARN] V3 cache not found at {v3_dir}, will regenerate HDF5 from CSV')

# 3. 写入 V5 配置
v5 = """# ============================================================
# V5: 控制变量实验 — 只改 PLE num_layers: 1 → 2
# ============================================================
# 基线: V3 (MMoE 0.686, PLE 0.680, SB 0.669)
# 唯一变化: PLE num_layers: 1 → 2
# 预期: PLE > MMoE > SharedBottom
# ============================================================

SharedBottom_KuaiRand1K:
    model: SharedBottom
    dataset_id: KuaiRand1K
    task_labels: ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'is_hate', 'is_profile_enter']
    loss: ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
    loss_weight: [1.5, 3.0, 5.0, 5.0, 5.0, 1.5, 3.0, 2.0]
    metrics: ['logloss', 'AUC', 'gAUC']
    task: ['binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification']
    num_tasks: 8
    optimizer: adam
    learning_rate: 5.e-4
    max_gradient_norm: 5.0
    bottom_hidden_units: [512, 512, 512]
    tower_hidden_units: [128, 128]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-6
    batch_norm: True
    net_dropout: 0.1
    batch_size: 4096
    embedding_dim: 16
    epochs: 40
    shuffle: True
    seed: 2023
    monitor: 'AUC'
    monitor_mode: 'max'
    group_id: 'user_id'
    model_root: './checkpoints'
    num_workers: 3
    verbose: 1
    early_stop_patience: 10
    save_best_only: True

MMoE_KuaiRand1K:
    model: MMoE
    dataset_id: KuaiRand1K
    task_labels: ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'is_hate', 'is_profile_enter']
    loss: ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
    loss_weight: [1.5, 3.0, 5.0, 5.0, 5.0, 1.5, 3.0, 2.0]
    metrics: ['logloss', 'AUC', 'gAUC']
    task: ['binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification']
    num_tasks: 8
    optimizer: adam
    learning_rate: 5.e-4
    max_gradient_norm: 5.0
    num_experts: 8
    expert_hidden_units: [512, 256, 128]
    gate_hidden_units: [128, 64]
    tower_hidden_units: [128, 64]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-6
    batch_norm: True
    net_dropout: 0.1
    batch_size: 4096
    embedding_dim: 16
    epochs: 40
    shuffle: True
    seed: 2023
    monitor: 'AUC'
    monitor_mode: 'max'
    group_id: 'user_id'
    model_root: './checkpoints'
    num_workers: 3
    verbose: 1
    early_stop_patience: 10
    save_best_only: True

PLE_KuaiRand1K:
    model: PLE
    dataset_id: KuaiRand1K
    task_labels: ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'is_hate', 'is_profile_enter']
    loss: ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
    loss_weight: [1.5, 3.0, 5.0, 5.0, 5.0, 1.5, 3.0, 2.0]
    metrics: ['logloss', 'AUC', 'gAUC']
    task: ['binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification', 'binary_classification']
    num_tasks: 8
    optimizer: adam
    learning_rate: 5.e-4
    max_gradient_norm: 5.0
    # ===== V5 唯一变化: num_layers 1 → 2 =====
    num_layers: 2
    num_shared_experts: 4
    num_specific_experts: 2
    # ===== 其他与 V3 完全一致 =====
    expert_hidden_units: [512, 256, 128]
    gate_hidden_units: [128, 64]
    tower_hidden_units: [128, 64]
    hidden_activations: relu
    net_regularizer: 0
    embedding_regularizer: 1.e-6
    batch_norm: True
    net_dropout: 0.1
    batch_size: 4096
    embedding_dim: 16
    epochs: 40
    shuffle: True
    seed: 2023
    monitor: 'AUC'
    monitor_mode: 'max'
    group_id: 'user_id'
    model_root: './checkpoints'
    num_workers: 3
    verbose: 1
    early_stop_patience: 10
    save_best_only: True
"""

config_path = f'{CONFIG_DIR}/model_config_v5.yaml'
with open(config_path, 'w') as f:
    f.write(v5)
print(f'[OK] V5 config -> {config_path}')

# 4. 激活 V5
shutil.copy2(config_path, f'{CONFIG_DIR}/model_config.yaml')
print('[OK] model_config.yaml -> V5')

print('\n' + '='*60)
print('V5 控制变量实验 已就绪')
print('='*60)
print(f'基线: V3 (MMoE=0.686, PLE=0.680, SB=0.669)')
print(f'唯一变化: PLE num_layers 1 → 2')
print(f'缓存: 复用 V3 (embedding_dim=16)')
print(f'\n训练命令:')
print(f'  /root/miniconda3/bin/python run_with_log.py SharedBottom_KuaiRand1K ./config 0 v5')
print(f'  /root/miniconda3/bin/python run_with_log.py MMoE_KuaiRand1K ./config 0 v5')
print(f'  /root/miniconda3/bin/python run_with_log.py PLE_KuaiRand1K ./config 0 v5')
print('='*60)