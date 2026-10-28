#!/usr/bin/env python3
"""
方案 B + D：层级偏好 relevance + MLP 融合
===========================================
内存优化版：逐模型处理，用完即释放
"""

import os, sys, json, pickle, gc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# 强制刷新输出
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

PROJECT_ROOT = '/root/mtl4rec'
PRED_DIR = os.path.join(PROJECT_ROOT, 'results/predictions')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results/analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASK_NAMES = ['is_click', 'is_like', 'is_follow', 'is_comment',
              'is_forward', 'long_view', 'is_hate', 'is_profile_enter']
K_VALUES = [10, 20, 30, 50]

# ============================================================
# 方案 D: 层级偏好 Relevance
# ============================================================
def compute_relevance_hierarchical(y_true_dict):
    n = len(y_true_dict[TASK_NAMES[0]])
    pos_count = np.zeros(n, dtype=np.float32)
    for task in TASK_NAMES:
        if task == 'is_hate':
            continue
        pos_count += y_true_dict[task].astype(np.float32)
    
    has_follow_or_comment = (y_true_dict['is_follow'] + y_true_dict['is_comment']) > 0
    has_like_or_forward_or_long = (y_true_dict['is_like'] + y_true_dict['is_forward'] + y_true_dict['long_view']) > 0
    has_click_or_profile = (y_true_dict['is_click'] + y_true_dict['is_profile_enter']) > 0
    has_hate = y_true_dict['is_hate'] > 0
    
    relevance = np.where(has_follow_or_comment, 300,
                np.where(has_like_or_forward_or_long, 200,
                np.where(has_click_or_profile, 100, 0)))
    relevance = relevance.astype(np.float32) + pos_count
    relevance = np.where(has_hate, 0, relevance)
    return relevance

def compute_relevance_original(y_true_dict):
    weights = {
        'is_click': 1.0, 'is_like': 2.0, 'is_follow': 3.0,
        'is_comment': 3.0, 'is_forward': 2.0, 'long_view': 3.0,
        'is_hate': -2.0, 'is_profile_enter': 1.0,
    }
    n = len(y_true_dict[TASK_NAMES[0]])
    relevance = np.zeros(n, dtype=np.float32)
    for task, weight in weights.items():
        relevance += weight * y_true_dict[task].astype(np.float32)
    return np.maximum(relevance, 0)

def compute_ndcg_at_k(y_true_dict, scores, k=10, relevance_fn=None):
    if relevance_fn is None:
        relevance_fn = compute_relevance_hierarchical
    n = len(scores)
    if n == 0:
        return 0.0
    relevance = relevance_fn(y_true_dict)
    ranking = np.argsort(-scores)
    dcg = sum(float(relevance[ranking[i]]) / np.log2(i + 2) for i in range(min(k, n)))
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = sum(float(ideal_relevance[i]) / np.log2(i + 2) for i in range(min(k, n)))
    return dcg / (idcg + 1e-8)

# ============================================================
# 方案 B: MLP 融合
# ============================================================
class FusionMLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp_fusion(y_pred, y_true, relevance_fn, n_samples, epochs=50, lr=0.001):
    """内存优化版 MLP 训练"""
    # 提取预测概率 (float32 节省内存)
    X = np.column_stack([y_pred[t].astype(np.float32) for t in TASK_NAMES])
    y = relevance_fn(y_true).astype(np.float32)
    
    # 80/20 划分
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    del X, y  # 释放内存
    gc.collect()
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    
    # 转 tensor (float32)
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).float()
    del X_train, X_val, y_train, y_val
    gc.collect()
    
    model = FusionMLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    
    # 全量预测 (分批处理节省内存)
    X_all = np.column_stack([y_pred[t].astype(np.float32) for t in TASK_NAMES])
    X_all = scaler.transform(X_all).astype(np.float32)
    
    scores = np.zeros(n_samples, dtype=np.float32)
    batch_size = 50000
    model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = torch.from_numpy(X_all[i:end])
            scores[i:end] = model(batch).numpy()
    
    del X_all, X_train_t, X_val_t, y_train_t, y_val_t
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return scores, best_val_loss

def apply_manual_fusion(predictions, fusion_weights):
    scores = np.zeros(predictions['n_samples'], dtype=np.float32)
    for task, weight in fusion_weights.items():
        if weight == 0:
            continue
        p = np.clip(predictions['y_pred'][task].astype(np.float32), 1e-8, 1 - 1e-8)
        if weight > 0:
            scores += weight * np.log(p)
        else:
            scores += weight * np.log(1 - p)
    return scores

FUSION_STRATEGIES = {
    '仅点击': {'is_click': 1.0, 'is_like': 0, 'is_follow': 0, 'is_comment': 0,
               'is_forward': 0, 'long_view': 0, 'is_hate': 0, 'is_profile_enter': 0},
    '互动深度加权': {'is_click': 1.0, 'is_like': 2.0, 'is_follow': 3.0, 'is_comment': 3.0,
                    'is_forward': 2.0, 'long_view': 3.0, 'is_hate': -2.0, 'is_profile_enter': 1.0},
    '降负反馈综合': {'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 1.0,
                    'is_forward': 1.0, 'long_view': 1.0, 'is_hate': -1.0, 'is_profile_enter': 1.0},
}

def process_model(model_name, pred_file):
    """处理单个模型，返回结果并释放内存"""
    print(f"\n{'='*70}", flush=True)
    print(f"  [{model_name}]", flush=True)
    print(f"{'='*70}", flush=True)
    
    with open(os.path.join(PRED_DIR, pred_file), 'rb') as f:
        predictions = pickle.load(f)
    print(f"  样本数: {predictions['n_samples']:,}", flush=True)
    
    model_results = {}
    
    # ---- D: 层级 relevance 下的手工融合 NDCG ----
    print(f"\n  [D] 层级 relevance + 手工融合:", flush=True)
    d_results = {}
    for strategy_name, fusion_weights in FUSION_STRATEGIES.items():
        scores = apply_manual_fusion(predictions, fusion_weights)
        ndcg_vals = {}
        for k in K_VALUES:
            ndcg = compute_ndcg_at_k(predictions['y_true'], scores, k,
                                     relevance_fn=compute_relevance_hierarchical)
            ndcg_vals[f'NDCG@{k}'] = float(ndcg)
        d_results[strategy_name] = ndcg_vals
        print(f"    {strategy_name:12s}: NDCG@10={ndcg_vals['NDCG@10']:.4f}  "
              f"NDCG@20={ndcg_vals['NDCG@20']:.4f}  "
              f"NDCG@30={ndcg_vals['NDCG@30']:.4f}  "
              f"NDCG@50={ndcg_vals['NDCG@50']:.4f}", flush=True)
        del scores; gc.collect()
    model_results['hierarchical_relevance'] = d_results
    
    # ---- 对比: 原始数值 relevance 下的手工融合 NDCG ----
    print(f"\n  [对比] 原始数值 relevance + 手工融合:", flush=True)
    orig_results = {}
    for strategy_name, fusion_weights in FUSION_STRATEGIES.items():
        scores = apply_manual_fusion(predictions, fusion_weights)
        ndcg_vals = {}
        for k in K_VALUES:
            ndcg = compute_ndcg_at_k(predictions['y_true'], scores, k,
                                     relevance_fn=compute_relevance_original)
            ndcg_vals[f'NDCG@{k}'] = float(ndcg)
        orig_results[strategy_name] = ndcg_vals
        print(f"    {strategy_name:12s}: NDCG@10={ndcg_vals['NDCG@10']:.4f}  "
              f"NDCG@20={ndcg_vals['NDCG@20']:.4f}  "
              f"NDCG@30={ndcg_vals['NDCG@30']:.4f}  "
              f"NDCG@50={ndcg_vals['NDCG@50']:.4f}", flush=True)
        del scores; gc.collect()
    model_results['original_relevance'] = orig_results
    
    # ---- B: MLP 融合 (层级 relevance 作为训练目标) ----
    print(f"\n  [B] MLP 融合 (训练目标: 层级 relevance)...", flush=True)
    mlp_scores, val_loss = train_mlp_fusion(
        predictions['y_pred'], predictions['y_true'],
        compute_relevance_hierarchical, predictions['n_samples'])
    
    mlp_ndcg = {}
    for k in K_VALUES:
        ndcg = compute_ndcg_at_k(predictions['y_true'], mlp_scores, k,
                                 relevance_fn=compute_relevance_hierarchical)
        mlp_ndcg[f'NDCG@{k}'] = float(ndcg)
    model_results['mlp_fusion'] = mlp_ndcg
    print(f"    MLP验证loss: {val_loss:.4f}", flush=True)
    print(f"    MLP融合:   NDCG@10={mlp_ndcg['NDCG@10']:.4f}  "
          f"NDCG@20={mlp_ndcg['NDCG@20']:.4f}  "
          f"NDCG@30={mlp_ndcg['NDCG@30']:.4f}  "
          f"NDCG@50={mlp_ndcg['NDCG@50']:.4f}", flush=True)
    del mlp_scores; gc.collect()
    
    # ---- 对比: 原始 relevance 下 MLP 融合 ----
    print(f"\n  [B对比] MLP 融合 (训练目标: 原始数值 relevance)...", flush=True)
    mlp_scores_orig, val_loss_orig = train_mlp_fusion(
        predictions['y_pred'], predictions['y_true'],
        compute_relevance_original, predictions['n_samples'])
    
    mlp_ndcg_orig = {}
    for k in K_VALUES:
        ndcg = compute_ndcg_at_k(predictions['y_true'], mlp_scores_orig, k,
                                 relevance_fn=compute_relevance_original)
        mlp_ndcg_orig[f'NDCG@{k}'] = float(ndcg)
    model_results['mlp_fusion_original'] = mlp_ndcg_orig
    print(f"    MLP验证loss: {val_loss_orig:.4f}", flush=True)
    print(f"    MLP融合(原): NDCG@10={mlp_ndcg_orig['NDCG@10']:.4f}  "
          f"NDCG@20={mlp_ndcg_orig['NDCG@20']:.4f}  "
          f"NDCG@30={mlp_ndcg_orig['NDCG@30']:.4f}  "
          f"NDCG@50={mlp_ndcg_orig['NDCG@50']:.4f}", flush=True)
    del mlp_scores_orig; gc.collect()
    
    # ---- 汇总 ----
    best_manual_hr = max(d_results.items(), key=lambda x: x[1]['NDCG@10'])
    best_manual_or = max(orig_results.items(), key=lambda x: x[1]['NDCG@10'])
    print(f"\n  [汇总] 最佳手工融合 vs MLP 融合:", flush=True)
    print(f"    层级relevance下: 最佳手工={best_manual_hr[0]}({best_manual_hr[1]['NDCG@10']:.4f}) "
          f"vs MLP({mlp_ndcg['NDCG@10']:.4f})", flush=True)
    print(f"    数值relevance下: 最佳手工={best_manual_or[0]}({best_manual_or[1]['NDCG@10']:.4f}) "
          f"vs MLP({mlp_ndcg_orig['NDCG@10']:.4f})", flush=True)
    
    # 释放 predictions
    del predictions
    gc.collect()
    
    return model_results

def main():
    print("=" * 70, flush=True)
    print("  方案 B + D：层级偏好 relevance + MLP 融合 (内存优化版)", flush=True)
    print("=" * 70, flush=True)
    
    models = {
        'SharedBottom': 'SharedBottom_base_predictions.pkl',
        'MMoE_v3': 'MMoE_v3_predictions.pkl',
        'PLE_v7': 'PLE_v7_predictions.pkl',
    }
    
    all_results = {}
    
    for model_name, pred_file in models.items():
        all_results[model_name] = process_model(model_name, pred_file)
    
    # ============================================================
    # 汇总对比表
    # ============================================================
    print(f"\n\n{'='*70}", flush=True)
    print(f"  NDCG@10 汇总对比", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'模型':<16} {'层级+仅点击':>10} {'层级+深度加权':>10} {'层级+降负反馈':>10} {'层级+MLP':>10} | "
          f"{'数值+仅点击':>10} {'数值+深度加权':>10} {'数值+降负反馈':>10} {'数值+MLP':>10}", flush=True)
    print(f"{'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10} | {'-'*10} {'-'*10} {'-'*10} {'-'*10}", flush=True)
    
    for model_name in models:
        r = all_results[model_name]
        h = r['hierarchical_relevance']
        o = r['original_relevance']
        print(f"{model_name:<16} "
              f"{h['仅点击']['NDCG@10']:>10.4f} {h['互动深度加权']['NDCG@10']:>10.4f} "
              f"{h['降负反馈综合']['NDCG@10']:>10.4f} {r['mlp_fusion']['NDCG@10']:>10.4f} | "
              f"{o['仅点击']['NDCG@10']:>10.4f} {o['互动深度加权']['NDCG@10']:>10.4f} "
              f"{o['降负反馈综合']['NDCG@10']:>10.4f} {r['mlp_fusion_original']['NDCG@10']:>10.4f}", flush=True)
    
    # 保存结果
    result_path = os.path.join(OUTPUT_DIR, 'bd_fusion_results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  结果已保存: {result_path}", flush=True)

if __name__ == '__main__':
    main()