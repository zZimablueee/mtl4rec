#!/usr/bin/env python3
"""验证 NDCG 计算是否正确，分析 0.695 是否合理"""
import pickle, numpy as np, pandas as pd

TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment',
         'is_forward', 'long_view', 'is_hate', 'is_profile_enter']

# 加载 PLE_v7 预测
with open('/root/mtl4rec/results/predictions/PLE_v7_predictions.pkl', 'rb') as f:
    pred = pickle.load(f)

print(f"Total samples: {pred['n_samples']}")

# 层级 relevance
def compute_relevance_hierarchical(y_true_dict):
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
        if is_L3: max_tier = 3
        elif is_L2: max_tier = 2
        elif is_L1: max_tier = 1
        else: max_tier = 0
        all_pos = tier_L3 + tier_L2 + tier_L1
        n_pos = sum(1 for t in all_pos if y_true_dict[t][i] == 1)
        bonus = 1 if n_pos >= 2 else 0
        relevance[i] = max_tier + bonus
    return relevance

rel = compute_relevance_hierarchical(pred['y_true'])

# 分布
print("\n=== Relevance 分布 ===")
for v in sorted(set(rel)):
    count = (rel == v).sum()
    print(f"  rel={v:.0f}: {count} ({100*count/len(rel):.1f}%)")

# IDCG@10 分析
ideal = np.sort(rel)[::-1]
print(f"\n=== IDCG@10 分析 ===")
print(f"Top 10 ideal relevance: {ideal[:10]}")
idcg_10 = sum(ideal[i] / np.log2(i + 2) for i in range(10))
print(f"IDCG@10 = {idcg_10:.4f}")

# 如果随机排序
np.random.seed(42)
random_scores = np.random.randn(len(rel))
ranking_r = np.argsort(-random_scores)
dcg_r = sum(rel[ranking_r[i]] / np.log2(i + 2) for i in range(10))
ndcg_r = dcg_r / (idcg_10 + 1e-8)
print(f"\n随机排序 NDCG@10 = {ndcg_r:.4f} (baseline)")

# PLE_v7 仅点击策略
p = np.clip(pred['y_pred']['is_click'], 1e-8, 1 - 1e-8)
scores = np.log(p)

ranking = np.argsort(-scores)
dcg_10 = sum(rel[ranking[i]] / np.log2(i + 2) for i in range(10))
ndcg_10 = dcg_10 / (idcg_10 + 1e-8)

print(f"\n=== PLE_v7 仅点击 NDCG@10 ===")
print(f"DCG@10  = {dcg_10:.4f}")
print(f"NDCG@10 = {ndcg_10:.4f}")
print(f"\nTop 10 排名详情:")
for i, idx in enumerate(ranking[:10]):
    labels = {t: int(pred['y_true'][t][idx]) for t in TASKS if pred['y_true'][t][idx] == 1}
    print(f"  Rank {i+1}: idx={idx}, rel={rel[idx]:.0f}, p_click={p[idx]:.4f}, labels={labels}")

# gNDCG（用户级）
df = pd.read_csv('/root/mtl4rec/data/KuaiRand1K/test.csv', usecols=['user_id'])
test_user_ids = df['user_id'].values
unique_users = np.unique(test_user_ids)
print(f"\n=== gNDCG@10（用户级）===")
print(f"用户数: {len(unique_users)}")

ndcgs = []
for u in unique_users:
    mask = test_user_ids == u
    if mask.sum() < 10:
        continue
    rel_u = rel[mask]
    scores_u = scores[mask]
    ranking_u = np.argsort(-scores_u)
    dcg_u = sum(rel_u[ranking_u[i]] / np.log2(i + 2) for i in range(10))
    ideal_u = np.sort(rel_u)[::-1]
    idcg_u = sum(ideal_u[i] / np.log2(i + 2) for i in range(10))
    ndcg_u = dcg_u / (idcg_u + 1e-8)
    ndcgs.append(ndcg_u)

gndcg = np.mean(ndcgs)
print(f"gNDCG@10 = {gndcg:.4f}")
print(f"NDCG per user: min={min(ndcgs):.4f}, max={max(ndcgs):.4f}, std={np.std(ndcgs):.4f}")

# 分析: 为什么全局 NDCG 高而 gNDCG 低
print(f"\n=== 关键分析 ===")
print(f"全局 NDCG@10 高的原因:")
print(f"  1. IDCG@10 = {idcg_10:.2f}（基于 665K 样本的 top 10 ideal）")
print(f"  2. 只需在 665K 样本中找出 10 个高 relevance 的")
print(f"  3. 有 {int((rel==3).sum())} 个 rel=3 的样本，找 10 个很容易")
print(f"")
print(f"gNDCG@10 更合理的原因:")
print(f"  1. 每个用户独立计算 NDCG，再取平均")
print(f"  2. 用户级别的 IDCG 通常更小（一个用户可能没有高 relevance 样本）")
print(f"  3. gNDCG@10 = {gndcg:.4f} 是更真实的排序能力指标")

# 结论
print(f"\n{'='*60}")
print(f"结论: 全局 NDCG@10 = {ndcg_10:.4f} 不是计算错误，而是全局排名导致的")
print(f"真实的用户级排序能力: gNDCG@10 = {gndcg:.4f}")
print(f"建议: 报告中以 gNDCG 为主要指标，全局 NDCG 作为参考")
print(f"{'='*60}")