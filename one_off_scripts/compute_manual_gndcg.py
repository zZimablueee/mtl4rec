"""
计算手工融合策略的 gNDCG@10 值，补全 bd_fusion_output.log 中的对比表。
"""
import numpy as np
import pickle
import pandas as pd
import sys, os
sys.path.insert(0, '/root/mtl4rec')
sys.path.insert(0, '/root/STEM/fuxictr')

TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
         'long_view', 'is_hate', 'is_profile_enter']

# ============================================================
# 方案 D: 层级 relevance
# ============================================================
def compute_relevance_hierarchical(y_true_dict):
    """层级偏好 relevance: 关注/评论 > 点赞/转发/有效播放 > 点击/主页 > 负反馈惩罚"""
    n = len(y_true_dict[TASKS[0]])
    relevance = np.zeros(n)
    tier_L3 = ['is_follow', 'is_comment']      # 强互动
    tier_L2 = ['is_like', 'is_forward', 'long_view']  # 中互动
    tier_L1 = ['is_click', 'is_profile_enter']  # 弱互动

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

# ============================================================
# 数值 relevance (原始方案)
# ============================================================
def compute_relevance_numerical(y_true_dict):
    """数值权重: 点击10 + 点赞20 + 关注30 + 评论30 + 转发30 + 有效播放20 + 主页15 - hate*50"""
    relevance = np.zeros(len(y_true_dict[TASKS[0]]))
    relevance += y_true_dict['is_click'] * 10
    relevance += y_true_dict['is_like'] * 20
    relevance += y_true_dict['is_follow'] * 30
    relevance += y_true_dict['is_comment'] * 30
    relevance += y_true_dict['is_forward'] * 30
    relevance += y_true_dict['long_view'] * 20
    relevance += y_true_dict['is_profile_enter'] * 15
    relevance -= y_true_dict['is_hate'] * 50
    return np.clip(relevance, 0, None)

# ============================================================
# gNDCG 计算
# ============================================================
def compute_gndcg(rel, scores, user_ids, k=10):
    unique_users = np.unique(user_ids)
    ndcgs = []
    for u in unique_users:
        mask = user_ids == u
        if mask.sum() < k:
            continue
        rel_u = rel[mask]
        scores_u = scores[mask]
        ranking_u = np.argsort(-scores_u)
        dcg_u = sum(rel_u[ranking_u[i]] / np.log2(i + 2) for i in range(k))
        ideal_u = np.sort(rel_u)[::-1]
        idcg_u = sum(ideal_u[i] / np.log2(i + 2) for i in range(k))
        ndcg_u = dcg_u / (idcg_u + 1e-8)
        ndcgs.append(ndcg_u)
    return np.mean(ndcgs)

# ============================================================
# 手工融合策略
# ============================================================
def manual_fusion_click_only(predictions):
    """仅点击"""
    return predictions['y_pred']['is_click']

def manual_fusion_weighted(predictions):
    """互动深度加权"""
    scores = np.zeros_like(predictions['y_pred']['is_click'])
    scores += predictions['y_pred']['is_click'] * 1.0
    scores += predictions['y_pred']['is_like'] * 5.0
    scores += predictions['y_pred']['is_follow'] * 10.0
    scores += predictions['y_pred']['is_comment'] * 10.0
    scores += predictions['y_pred']['is_forward'] * 10.0
    scores += predictions['y_pred']['long_view'] * 2.0
    scores += predictions['y_pred']['is_profile_enter'] * 3.0
    scores -= predictions['y_pred']['is_hate'] * 10.0
    return scores

# ============================================================
# 加载预测并计算
# ============================================================
MODELS = {
    'SharedBottom': '/root/mtl4rec/results/predictions/SharedBottom_base_predictions.pkl',
    'MMoE_v3': '/root/mtl4rec/results/predictions/MMoE_v3_predictions.pkl',
    'PLE_v7': '/root/mtl4rec/results/predictions/PLE_v7_predictions.pkl',
}

print("=" * 70)
print("  手工融合策略 gNDCG@10 对比")
print("=" * 70)
print()

# 加载 test_user_ids 从 test.csv
test_df = pd.read_csv('/root/mtl4rec/data/KuaiRand1K/test.csv')
test_user_ids = test_df['user_id'].values

print(f"测试集样本数: {len(test_user_ids)}")
print(f"测试集用户数: {len(np.unique(test_user_ids))}")
print()

results_numerical = {}
results_hierarchical = {}

for model_name, pred_path in MODELS.items():
    print(f"--- {model_name} ---")
    with open(pred_path, 'rb') as f:
        pred = pickle.load(f)

    predictions = {'y_pred': pred['y_pred'], 'y_true': pred['y_true']}

    # 策略1: 仅点击
    scores_click = manual_fusion_click_only(predictions)
    rel_num = compute_relevance_numerical(predictions['y_true'])
    rel_hier = compute_relevance_hierarchical(predictions['y_true'])

    gndcg_click_num = compute_gndcg(rel_num, scores_click, test_user_ids, k=10)
    gndcg_click_hier = compute_gndcg(rel_hier, scores_click, test_user_ids, k=10)

    # 策略2: 加权融合
    scores_weighted = manual_fusion_weighted(predictions)
    gndcg_weighted_num = compute_gndcg(rel_num, scores_weighted, test_user_ids, k=10)
    gndcg_weighted_hier = compute_gndcg(rel_hier, scores_weighted, test_user_ids, k=10)

    results_numerical[model_name] = {
        'click_only': gndcg_click_num,
        'weighted': gndcg_weighted_num,
    }
    results_hierarchical[model_name] = {
        'click_only': gndcg_click_hier,
        'weighted': gndcg_weighted_hier,
    }

    print(f"  数值 relevance: 仅点击 gNDCG@10={gndcg_click_num:.4f}, 加权 gNDCG@10={gndcg_weighted_num:.4f}")
    print(f"  层级 relevance: 仅点击 gNDCG@10={gndcg_click_hier:.4f}, 加权 gNDCG@10={gndcg_weighted_hier:.4f}")
    print()

# 汇总
print("=" * 70)
print("  汇总对比表")
print("=" * 70)
print()
print(f"{'策略':<40} {'SharedBottom':>13} {'MMoE_v3':>13} {'PLE_v7':>13}")
print("-" * 80)

# 数值 relevance
print(f"{'手工融合 + 数值relevance (仅点击)':<40} "
      f"{results_numerical['SharedBottom']['click_only']:>13.4f} "
      f"{results_numerical['MMoE_v3']['click_only']:>13.4f} "
      f"{results_numerical['PLE_v7']['click_only']:>13.4f}")

print(f"{'手工融合 + 数值relevance (加权)':<40} "
      f"{results_numerical['SharedBottom']['weighted']:>13.4f} "
      f"{results_numerical['MMoE_v3']['weighted']:>13.4f} "
      f"{results_numerical['PLE_v7']['weighted']:>13.4f}")

# 层级 relevance
print(f"{'手工融合 + 层级relevance (仅点击)':<40} "
      f"{results_hierarchical['SharedBottom']['click_only']:>13.4f} "
      f"{results_hierarchical['MMoE_v3']['click_only']:>13.4f} "
      f"{results_hierarchical['PLE_v7']['click_only']:>13.4f}")

print(f"{'手工融合 + 层级relevance (加权)':<40} "
      f"{results_hierarchical['SharedBottom']['weighted']:>13.4f} "
      f"{results_hierarchical['MMoE_v3']['weighted']:>13.4f} "
      f"{results_hierarchical['PLE_v7']['weighted']:>13.4f}")

print()
print("=" * 70)
print("  关键发现")
print("=" * 70)
print()
print("gNDCG (用户级 NDCG) 是推荐系统文献中的标准指标。")
print("它先对每个用户独立计算 NDCG，再对所有用户取平均，")
print("反映了模型「为每个用户排序」的真实能力。")
print()
print("全局 NDCG 高是因为在 665K 样本中找 10 个高 relevance 的很容易——")
print("模型只需把热门的、高互动的 item 排到前面，不需要关心是哪个用户的。")
print("gNDCG 则要求模型对每个用户都排好序，这才是推荐系统的真实挑战。")