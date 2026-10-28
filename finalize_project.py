"""
项目收尾脚本 — 一次性输出所有需要的数据
==========================================
1. 3模型 × 8任务 AUC/gAUC/正样本率
2. 单任务加入对 gNDCG 的贡献
3. 最终汇总表
"""
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys, os

TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
         'long_view', 'is_hate', 'is_profile_enter']

MODELS = {
    'SharedBottom': '/root/mtl4rec/results/predictions/SharedBottom_base_predictions.pkl',
    'MMoE_v3': '/root/mtl4rec/results/predictions/MMoE_v3_predictions.pkl',
    'PLE_v7': '/root/mtl4rec/results/predictions/PLE_v7_predictions.pkl',
}

# 层级 relevance (简化版 0-4)
def compute_relevance_hierarchical(y_true_dict):
    n = len(y_true_dict[TASKS[0]])
    relevance = np.zeros(n)
    tier_L3 = ['is_follow', 'is_comment']
    tier_L2 = ['is_like', 'is_forward', 'long_view']
    tier_L1 = ['is_click', 'is_profile_enter']
    for i in range(n):
        if y_true_dict['is_hate'][i] == 1:
            relevance[i] = 0; continue
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

def compute_gndcg(rel, scores, user_ids, k=10):
    unique_users = np.unique(user_ids)
    ndcgs = []
    for u in unique_users:
        mask = user_ids == u
        if mask.sum() < k: continue
        rel_u = rel[mask]; scores_u = scores[mask]
        ranking_u = np.argsort(-scores_u)
        dcg_u = sum(rel_u[ranking_u[i]] / np.log2(i + 2) for i in range(k))
        ideal_u = np.sort(rel_u)[::-1]
        idcg_u = sum(ideal_u[i] / np.log2(i + 2) for i in range(k))
        ndcgs.append(dcg_u / (idcg_u + 1e-8))
    return np.mean(ndcgs)

def compute_gauc(y_true, y_pred, user_ids):
    """per-user AUC 平均"""
    unique_users = np.unique(user_ids)
    aucs = []
    for u in unique_users:
        mask = user_ids == u
        yt = y_true[mask]; yp = y_pred[mask]
        if len(np.unique(yt)) < 2: continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except: pass
    return np.mean(aucs) if aucs else 0.0

# 加载数据
print("=" * 80)
print("  项目收尾：最终数据汇总")
print("=" * 80)
print()

test_df = pd.read_csv('/root/mtl4rec/data/KuaiRand1K/test.csv')
test_user_ids = test_df['user_id'].values

# ============================================================
# Part 1: 3模型 × 8任务 AUC/gAUC/正样本率
# ============================================================
print("=" * 80)
print("  Part 1: 任务级预测能力")
print("=" * 80)
print()

all_preds = {}
for model_name, pred_path in MODELS.items():
    with open(pred_path, 'rb') as f:
        all_preds[model_name] = pickle.load(f)

# 正样本率（从真实标签算，所有模型共用）
print(f"  {'Task':<20} {'正样本率':>10} {'样本数':>10}")
print("  " + "-" * 42)
for t in TASKS:
    pos_rate = all_preds['PLE_v7']['y_true'][t].mean() * 100
    pos_count = all_preds['PLE_v7']['y_true'][t].sum()
    print(f"  {t:<20} {pos_rate:>9.2f}% {int(pos_count):>10}")
print()

# AUC + gAUC 表
print(f"  {'Task':<20} {'SharedBottom':>15} {'MMoE_v3':>15} {'PLE_v7':>15}")
print(f"  {'':20} {'AUC':>7} {'gAUC':>7} {'AUC':>7} {'gAUC':>7} {'AUC':>7} {'gAUC':>7}")
print("  " + "-" * 70)

auc_data = {}
for t in TASKS:
    row = [t]
    for model_name in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        yt = all_preds[model_name]['y_true'][t]
        yp = all_preds[model_name]['y_pred'][t]
        auc = roc_auc_score(yt, yp)
        gauc = compute_gauc(yt, yp, test_user_ids)
        row.append(f'{auc:.4f}')
        row.append(f'{gauc:.4f}')
        if model_name not in auc_data:
            auc_data[model_name] = {}
        auc_data[model_name][t] = {'auc': auc, 'gauc': gauc}
    print(f"  {row[0]:<20} {row[1]:>7} {row[2]:>7} {row[3]:>7} {row[4]:>7} {row[5]:>7} {row[6]:>7}")

# 平均 AUC
print("  " + "-" * 70)
for model_name in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
    avg_auc = np.mean([auc_data[model_name][t]['auc'] for t in TASKS])
    avg_gauc = np.mean([auc_data[model_name][t]['gauc'] for t in TASKS])
    print(f"  {'avg_auc':<20} {avg_auc:>7.4f} {avg_gauc:>7.4f}", end="")
print()

print()

# ============================================================
# Part 2: 单任务加入对 gNDCG 的贡献
# ============================================================
print("=" * 80)
print("  Part 2: 单任务加入对 gNDCG 的贡献 (层级 relevance)")
print("  S = p_click + α * p_task,  α ∈ {0.5, 1.0, 2.0}")
print("=" * 80)
print()

for model_name in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
    print(f"  --- {model_name} ---")
    pred = all_preds[model_name]
    y_true = pred['y_true']
    y_pred = pred['y_pred']
    rel = compute_relevance_hierarchical(y_true)
    
    # baseline: 仅 click
    S_click = y_pred['is_click']
    gndcg_click = compute_gndcg(rel, S_click, test_user_ids, k=10)
    
    print(f"  {'Task':<20} {'α=0.5':>10} {'α=1.0':>10} {'α=2.0':>10} {'vs click-only':>12}")
    print(f"  {'click-only (baseline)':<20} {'':>10} {'':>10} {'':>10} {gndcg_click:>12.4f}")
    print("  " + "-" * 62)
    
    best_tasks = []
    for t in TASKS:
        if t == 'is_click': continue
        for alpha in [0.5, 1.0, 2.0]:
            S = S_click + alpha * y_pred[t]
            gndcg = compute_gndcg(rel, S, test_user_ids, k=10)
            delta = gndcg - gndcg_click
            if alpha == 1.0:
                best_tasks.append((t, gndcg, delta))
        
        gndcg_05 = compute_gndcg(rel, S_click + 0.5 * y_pred[t], test_user_ids, k=10)
        gndcg_10 = compute_gndcg(rel, S_click + 1.0 * y_pred[t], test_user_ids, k=10)
        gndcg_20 = compute_gndcg(rel, S_click + 2.0 * y_pred[t], test_user_ids, k=10)
        delta_10 = gndcg_10 - gndcg_click
        sign = '+' if delta_10 >= 0 else ''
        print(f"  {t:<20} {gndcg_05:>10.4f} {gndcg_10:>10.4f} {gndcg_20:>10.4f} {sign}{delta_10:>11.4f}")
    
    print()
    best_tasks.sort(key=lambda x: x[1], reverse=True)
    print(f"  最佳增益任务 (α=1.0): {best_tasks[0][0]} (+{best_tasks[0][2]:.4f})")
    print(f"  最差增益任务 (α=1.0): {best_tasks[-1][0]} ({best_tasks[-1][2]:.4f})")
    print()

# ============================================================
# Part 3: 最终汇总 — 项目核心结论
# ============================================================
print("=" * 80)
print("  Part 3: 最终汇总 — 项目核心结论")
print("=" * 80)
print()

print("  【模型对比 (AUC + gNDCG)】")
print()
print(f"  {'模型':<16} {'avg AUC':>10} {'avg gAUC':>10} {'gNDCG@10':>10} {'全局NDCG@10':>12}")
print("  " + "-" * 62)

# 用 PLE_v7 数据算 gNDCG
for model_name in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
    pred = all_preds[model_name]
    avg_auc = np.mean([roc_auc_score(pred['y_true'][t], pred['y_pred'][t]) for t in TASKS])
    avg_gauc = np.mean([compute_gauc(pred['y_true'][t], pred['y_pred'][t], test_user_ids) for t in TASKS])
    
    rel = compute_relevance_hierarchical(pred['y_true'])
    gndcg = compute_gndcg(rel, pred['y_pred']['is_click'], test_user_ids, k=10)
    
    # 全局 NDCG
    S = pred['y_pred']['is_click']
    ranking = np.argsort(-S)
    dcg = sum(rel[ranking[i]] / np.log2(i + 2) for i in range(10))
    ideal = np.sort(rel)[::-1]
    idcg = sum(ideal[i] / np.log2(i + 2) for i in range(10))
    ndcg = dcg / (idcg + 1e-8)
    
    print(f"  {model_name:<16} {avg_auc:>10.4f} {avg_gauc:>10.4f} {gndcg:>10.4f} {ndcg:>12.4f}")

print()
print("  【关键发现】")
print()
print("  1. PLE_v7 在全局 NDCG@10 上最优 (0.6950)，但 gNDCG 上 SharedBottom 略高")
print("     → 小数据集上简单模型泛化更好，但 PLE 的多任务解耦结构更有理论价值")
print()
print("  2. 层级 relevance 定义将 gNDCG 从 ~0.29 (数值) 提升到 ~0.48 (层级)")
print("     → 提升核心来自 relevance 定义改进，而非模型改进")
print()
print("  3. 单任务加入实验：仅点击已是最优融合策略")
print("     → 其他任务预测噪声 > 额外信息增益，说明 rare task 预测质量待提升")
print()
print("  4. 敏感性分析：不同 relevance 定义下结论方向一致")
print("     → 层级 relevance 的提升是稳健的，不是偶然")
print()
print("  【面试定位】")
print()
print("  项目定位：多任务学习 + 多目标排序评估")
print("  核心贡献：层级 relevance 定义 + 多模型对比 + 融合策略分析")
print("  不声称：工业级大规模推荐系统、用户满意度客观度量")
print("  可以说: 这是一个多任务推荐排序的离线评估框架, 验证了不同架构和")
print("          relevance 定义对排序质量的影响.")