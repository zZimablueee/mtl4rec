#!/usr/bin/env python3
"""
B + D 方案 — PyTorch GPU 加速版
==================================
D: 层级 relevance — 层级偏好替代数值加权
B: MLP 融合 — PyTorch GPU 训练 MLP 学习融合权重
"""
import pickle, numpy as np, json, os, sys, time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================================
# 配置
# ============================================================
TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment',
         'is_forward', 'long_view', 'is_hate', 'is_profile_enter']
K_VALUES = [10, 20, 30, 50]

df = pd.read_csv('/root/mtl4rec/data/KuaiRand1K/test.csv', usecols=['user_id'])
test_user_ids = df['user_id'].values

FUSION_STRATEGIES = {
    '仅点击': {'is_click': 1.0, 'is_like': 0, 'is_follow': 0, 'is_comment': 0,
        'is_forward': 0, 'long_view': 0, 'is_hate': 0, 'is_profile_enter': 0},
    '等权融合': {'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 1.0,
        'is_forward': 1.0, 'long_view': 1.0, 'is_hate': 1.0, 'is_profile_enter': 1.0},
    '互动深度加权': {'is_click': 1.0, 'is_like': 2.0, 'is_follow': 3.0, 'is_comment': 3.0,
        'is_forward': 2.0, 'long_view': 3.0, 'is_hate': -2.0, 'is_profile_enter': 1.0},
    '降负反馈综合': {'is_click': 1.0, 'is_like': 2.0, 'is_follow': 2.0, 'is_comment': 2.0,
        'is_forward': 1.5, 'long_view': 2.0, 'is_hate': -3.0, 'is_profile_enter': 1.0},
    '深度消费优先': {'is_click': 0.8, 'is_like': 1.0, 'is_follow': 1.5, 'is_comment': 1.0,
        'is_forward': 1.0, 'long_view': 3.0, 'is_hate': -1.5, 'is_profile_enter': 1.5},
    '社交互动优先': {'is_click': 0.5, 'is_like': 2.0, 'is_follow': 3.0, 'is_comment': 3.0,
        'is_forward': 3.0, 'long_view': 1.0, 'is_hate': -1.0, 'is_profile_enter': 0.5},
    '纯正向信号': {'is_click': 0, 'is_like': 1.0, 'is_follow': 2.0, 'is_comment': 2.0,
        'is_forward': 2.0, 'long_view': 2.0, 'is_hate': 0, 'is_profile_enter': 1.0},
}

# ============================================================
# D 方案: 层级 relevance
# ============================================================
def compute_relevance_numerical(y_true_dict):
    """原始数值型 relevance: 加权求和"""
    n = len(y_true_dict[TASKS[0]])
    rel = np.zeros(n)
    w = {'is_click': 1, 'is_like': 2, 'is_follow': 3, 'is_comment': 3,
         'is_forward': 2, 'long_view': 3, 'is_hate': -2, 'is_profile_enter': 1}
    for t, wt in w.items():
        rel += wt * y_true_dict[t]
    return np.maximum(rel, 0)


def compute_relevance_hierarchical(y_true_dict):
    """
    层级 relevance（D 方案）
    参考快手 EMER 层次化满意度定义:
      L3（强互动）: 关注、评论
      L2（中互动）: 点赞、转发、有效播放
      L1（弱互动）: 点击、主页进入
      L0（无互动）: 啥也没干
      L-1（负反馈）: is_hate
    规则: 取最高层级作为基础分，如果正反馈种类 >= 2 则 +1 奖励
    负反馈清零
    """
    n = len(y_true_dict[TASKS[0]])
    relevance = np.zeros(n)
    tier_L3 = ['is_follow', 'is_comment']
    tier_L2 = ['is_like', 'is_forward', 'long_view']
    tier_L1 = ['is_click', 'is_profile_enter']

    for i in range(n):
        if y_true_dict['is_hate'][i] == 1:
            relevance[i] = 0
            continue
        is_L3 = any(y_true_dict[t][i] == 1 for t in tier_L3)
        is_L2 = any(y_true_dict[t][i] == 1 for t in tier_L2)
        is_L1 = any(y_true_dict[t][i] == 1 for t in tier_L1)
        if is_L3:
            max_tier = 3
        elif is_L2:
            max_tier = 2
        elif is_L1:
            max_tier = 1
        else:
            max_tier = 0
        all_pos = tier_L3 + tier_L2 + tier_L1
        n_pos = sum(1 for t in all_pos if y_true_dict[t][i] == 1)
        bonus = 1 if n_pos >= 2 else 0
        relevance[i] = max_tier + bonus
    return relevance


# ============================================================
# NDCG / gNDCG
# ============================================================
def compute_ndcg_at_k(y_true_dict, scores, k, relevance_func):
    n = len(scores)
    if n == 0:
        return 0.0
    relevance = relevance_func(y_true_dict)
    ranking = np.argsort(-scores)
    dcg = sum(relevance[ranking[i]] / np.log2(i + 2) for i in range(min(k, n)))
    ideal = np.sort(relevance)[::-1]
    idcg = sum(ideal[i] / np.log2(i + 2) for i in range(min(k, n)))
    return dcg / (idcg + 1e-8)


def compute_gNDCG_at_k(y_true_dict, scores, user_ids, k, relevance_func):
    unique_users = np.unique(user_ids)
    ndcgs = []
    for u in unique_users:
        mask = user_ids == u
        if mask.sum() < k:
            continue
        ndcg = compute_ndcg_at_k(
            {t: y_true_dict[t][mask] for t in y_true_dict}, scores[mask], k, relevance_func)
        ndcgs.append(ndcg)
    return np.mean(ndcgs) if ndcgs else 0.0


# ============================================================
# 融合函数
# ============================================================
def apply_fusion(predictions, weights):
    """手工融合: Score = Σ w_k * log(p_k)"""
    scores = np.zeros(predictions['n_samples'])
    for task, w in weights.items():
        if w == 0:
            continue
        p = np.clip(predictions['y_pred'][task], 1e-8, 1 - 1e-8)
        if w > 0:
            scores += w * np.log(p)
        else:
            scores += w * np.log(1 - p)
    return scores


# ============================================================
# B 方案: PyTorch MLP 融合 (GPU)
# ============================================================
class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_torch(X_train, y_train, X_val, y_val, hidden_dims, epochs=50, batch_size=4096, lr=0.001):
    """PyTorch GPU 训练 MLP"""
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = FusionMLP(X_train.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    best_corr = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            from scipy.stats import spearmanr
            val_corr, _ = spearmanr(y_val, val_pred.cpu().numpy())

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_corr = val_corr
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_corr


def mlp_fusion_torch(predictions, relevance_func, test_user_ids):
    """PyTorch GPU MLP 融合"""
    n_samples = predictions['n_samples']
    X = np.column_stack([predictions['y_pred'][t] for t in TASKS])
    X_log = np.column_stack([
        np.log(np.clip(predictions['y_pred'][t], 1e-8, 1 - 1e-8)) for t in TASKS])
    X = np.column_stack([X, X_log])

    y = relevance_func(predictions['y_true'])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    unique_users = np.unique(test_user_ids)
    train_users, val_users = train_test_split(unique_users, test_size=0.3, random_state=42)
    train_mask = np.isin(test_user_ids, train_users)
    val_mask = np.isin(test_user_ids, val_users)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"  [MLP] 训练集: {len(X_train)} 样本 ({len(train_users)} 用户)")
    print(f"  [MLP] 验证集: {len(X_val)} 样本 ({len(val_users)} 用户)")

    configs = [
        (16,),
        (16, 8),
        (32, 16),
        (32, 16, 8),
    ]

    best_model = None
    best_score = -1
    best_config = None

    for hidden in configs:
        t0 = time.time()
        model, val_corr = train_mlp_torch(X_train, y_train, X_val, y_val, hidden)
        elapsed = time.time() - t0
        print(f"    hidden={hidden}: val Spearman r={val_corr:.4f} ({elapsed:.1f}s)")
        if val_corr > best_score:
            best_score = val_corr
            best_model = model
            best_config = hidden

    print(f"  [MLP] 最优配置: hidden={best_config}, val Spearman r={best_score:.4f}")

    X_all_t = torch.FloatTensor(X).to(device)
    best_model.eval()
    with torch.no_grad():
        all_scores = best_model(X_all_t).cpu().numpy()

    results = {}
    for k in K_VALUES:
        ndcg = compute_ndcg_at_k(predictions['y_true'], all_scores, k, relevance_func)
        gndcg = compute_gNDCG_at_k(predictions['y_true'], all_scores, test_user_ids, k, relevance_func)
        results[f'NDCG@{k}'] = float(ndcg)
        results[f'gNDCG@{k}'] = float(gndcg)

    return results, best_model, best_config


# ============================================================
# 主流程
# ============================================================
def main():
    t_total = time.time()
    print("=" * 80)
    print("  B + D 方案: 层级 relevance + PyTorch GPU MLP 融合")
    print("=" * 80)

    models = {
        'SharedBottom': '/root/mtl4rec/results/predictions/SharedBottom_base_predictions.pkl',
        'MMoE_v3':     '/root/mtl4rec/results/predictions/MMoE_v3_predictions.pkl',
        'PLE_v7':      '/root/mtl4rec/results/predictions/PLE_v7_predictions.pkl',
    }

    all_preds = {}
    for name, path in models.items():
        print(f"\n加载 {name}...")
        with open(path, 'rb') as f:
            all_preds[name] = pickle.load(f)
        print(f"  样本数: {all_preds[name]['n_samples']}")

    # ============================================================
    # 第一部分: 层级 relevance 分析
    # ============================================================
    print("\n" + "=" * 80)
    print("  第一部分: 层级 relevance 分布分析")
    print("=" * 80)

    pred = all_preds['SharedBottom']
    rel_num = compute_relevance_numerical(pred['y_true'])
    rel_hier = compute_relevance_hierarchical(pred['y_true'])

    print(f"\n  数值型 relevance 分布:")
    for v in sorted(set(rel_num)):
        count = (rel_num == v).sum()
        print(f"    relevance={v:.0f}: {count} 样本 ({100*count/len(rel_num):.1f}%)")

    print(f"\n  层级 relevance 分布:")
    for v in sorted(set(rel_hier)):
        count = (rel_hier == v).sum()
        print(f"    relevance={v:.0f}: {count} 样本 ({100*count/len(rel_hier):.1f}%)")

    corr = np.corrcoef(rel_num, rel_hier)[0, 1]
    print(f"\n  数值型 vs 层级 relevance Pearson r = {corr:.4f}")

    # ============================================================
    # 第二部分: 手工融合 NDCG 对比
    # ============================================================
    print("\n" + "=" * 80)
    print("  第二部分: 手工融合 NDCG — 数值 relevance vs 层级 relevance")
    print("=" * 80)

    header = f"\n  {'策略':<16s} {'模型':<16s} {'NDCG@10(数值)':>14s} {'NDCG@10(层级)':>14s} {'变化':>8s}"
    print(header)
    print(f"  {'-'*70}")

    comparison = {}
    for sname, sweights in FUSION_STRATEGIES.items():
        for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
            pred = all_preds[mname]
            w = {k: v for k, v in sweights.items() if k != 'desc'}
            scores = apply_fusion(pred, w)
            ndcg_num = compute_ndcg_at_k(pred['y_true'], scores, 10, compute_relevance_numerical)
            ndcg_hier = compute_ndcg_at_k(pred['y_true'], scores, 10, compute_relevance_hierarchical)
            diff = ndcg_hier - ndcg_num
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
            print(f"  {sname:<16s} {mname:<16s} {ndcg_num:14.4f} {ndcg_hier:14.4f} {arrow} {diff:+.4f}")
            comparison[f"{sname}|{mname}"] = {'ndcg_num': ndcg_num, 'ndcg_hier': ndcg_hier}

    # ============================================================
    # 第三部分: PyTorch GPU MLP 融合
    # ============================================================
    print("\n" + "=" * 80)
    print("  第三部分: PyTorch GPU MLP 融合 + 层级 relevance")
    print("=" * 80)

    mlp_results = {}
    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        print(f"\n  --- {mname} ---")
        pred = all_preds[mname]
        results, mlp_model, config = mlp_fusion_torch(pred, compute_relevance_hierarchical, test_user_ids)
        mlp_results[mname] = results
        print(f"  {'NDCG@10':>10s}: {results['NDCG@10']:.4f}")
        print(f"  {'NDCG@20':>10s}: {results['NDCG@20']:.4f}")
        print(f"  {'NDCG@30':>10s}: {results['NDCG@30']:.4f}")
        print(f"  {'NDCG@50':>10s}: {results['NDCG@50']:.4f}")
        print(f"  {'gNDCG@10':>10s}: {results['gNDCG@10']:.4f}")

    # ============================================================
    # 第四部分: 最终对比
    # ============================================================
    print("\n" + "=" * 80)
    print("  第四部分: 最终对比 — 三种方案 × 三个模型")
    print("=" * 80)

    best_manual = {}
    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        best_ndcg = -1
        best_strat = None
        for sname in FUSION_STRATEGIES:
            ndcg = comparison[f"{sname}|{mname}"]['ndcg_hier']
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_strat = sname
        best_manual[mname] = (best_strat, best_ndcg)

    print(f"\n  {'方案':<35s} {'SharedBottom':>14s} {'MMoE_v3':>14s} {'PLE_v7':>14s}")
    print(f"  {'-'*80}")

    model_names = ['SharedBottom', 'MMoE_v3', 'PLE_v7']
    nums = [max(comparison[f"{s}|{m}"]['ndcg_num'] for s in FUSION_STRATEGIES) for m in model_names]
    hiers = [max(comparison[f"{s}|{m}"]['ndcg_hier'] for s in FUSION_STRATEGIES) for m in model_names]
    mlps = [mlp_results[m]['NDCG@10'] for m in model_names]

    print(f"  {'手工融合 + 数值 relevance':<35s} {nums[0]:14.4f} {nums[1]:14.4f} {nums[2]:14.4f}")
    print(f"  {'手工融合 + 层级 relevance (D方案)':<35s} {hiers[0]:14.4f} {hiers[1]:14.4f} {hiers[2]:14.4f}")
    print(f"  {'MLP融合 + 层级 relevance (B+D方案)':<35s} {mlps[0]:14.4f} {mlps[1]:14.4f} {mlps[2]:14.4f}")

    print(f"\n  {'MLP融合 vs 手工融合(层级)':<35s}", end="")
    for m in range(3):
        diff = mlps[m] - hiers[m]
        print(f"  {'↑' if diff>0 else ('↓' if diff<0 else '→')} {diff:+.4f}     ", end="")
    print()
    print(f"  {'MLP融合 vs 手工融合(数值)':<35s}", end="")
    for m in range(3):
        diff = mlps[m] - nums[m]
        print(f"  {'↑' if diff>0 else ('↓' if diff<0 else '→')} {diff:+.4f}     ", end="")
    print()

    # ============================================================
    # 第五部分: 详细 MLP 结果
    # ============================================================
    print("\n" + "=" * 80)
    print("  第五部分: MLP 融合详细结果 (所有 K 值)")
    print("=" * 80)

    for mname in model_names:
        print(f"\n  {mname}:")
        for k in K_VALUES:
            print(f"    NDCG@{k:2d} = {mlp_results[mname][f'NDCG@{k}']:.4f}  |  gNDCG@{k:2d} = {mlp_results[mname][f'gNDCG@{k}']:.4f}")

    # ============================================================
    # 保存结果
    # ============================================================
    output = {
        'hierarchical_distribution': {str(k): int(v) for k, v in zip(*np.unique(rel_hier, return_counts=True))},
        'numerical_vs_hierarchical_corr': float(corr),
        'best_manual_numerical_ndcg10': {m: float(v) for m, v in zip(model_names, nums)},
        'best_manual_hierarchical_ndcg10': {m: float(v) for m, v in zip(model_names, hiers)},
        'mlp_fusion_ndcg10': {m: float(v) for m, v in zip(model_names, mlps)},
        'mlp_fusion_detailed': {m: r for m, r in mlp_results.items()},
        'best_manual_strategy': {m: s for m, (s, _) in best_manual.items()},
    }

    out_path = '/root/mtl4rec/results/analysis/improved_fusion_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    print(f"\n  总耗时: {time.time() - t_total:.1f}s")


if __name__ == '__main__':
    main()