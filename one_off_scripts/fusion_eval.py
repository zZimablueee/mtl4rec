
import pickle, numpy as np, json
from sklearn.metrics import roc_auc_score

TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment',
         'is_forward', 'long_view', 'is_hate', 'is_profile_enter']

# 加载 test.csv 的 user_id 用于 gNDCG
import pandas as pd
df = pd.read_csv('/root/mtl4rec/data/KuaiRand1K/test.csv', usecols=['user_id'])
test_user_ids = df['user_id'].values

# ============================================================
# 融合策略定义
# 核心思想: score = Σ w_k * log(p_k)
# 正权重 = 希望排在前面, 负权重 = 希望排在后面
# ============================================================
FUSION_STRATEGIES = {
    # 基线
    '仅点击': {
        'is_click': 1.0, 'is_like': 0, 'is_follow': 0, 'is_comment': 0,
        'is_forward': 0, 'long_view': 0, 'is_hate': 0, 'is_profile_enter': 0,
        'desc': '只用点击概率，忽略其他7个任务'
    },
    # 全任务等权
    '等权融合': {
        'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 1.0,
        'is_forward': 1.0, 'long_view': 1.0, 'is_hate': 1.0, 'is_profile_enter': 1.0,
        'desc': '8个任务等权相加，不考虑任务特性'
    },
    # 按互动深度加权
    '互动深度加权': {
        'is_click': 1.0, 'is_like': 2.0, 'is_follow': 3.0, 'is_comment': 3.0,
        'is_forward': 2.0, 'long_view': 3.0, 'is_hate': -2.0, 'is_profile_enter': 1.0,
        'desc': '点击<点赞<关注/评论/有效播放，负反馈惩罚'
    },
    # 降负反馈 + 互动优先
    '降负反馈综合': {
        'is_click': 1.0, 'is_like': 2.0, 'is_follow': 2.0, 'is_comment': 2.0,
        'is_forward': 1.5, 'long_view': 2.0, 'is_hate': -3.0, 'is_profile_enter': 1.0,
        'desc': '负反馈权重-3，强烈惩罚；互动任务权重2'
    },
    # 深度消费优先
    '深度消费优先': {
        'is_click': 0.8, 'is_like': 1.0, 'is_follow': 1.5, 'is_comment': 1.0,
        'is_forward': 1.0, 'long_view': 3.0, 'is_hate': -1.5, 'is_profile_enter': 1.5,
        'desc': '有效播放权重3x，强调用户真正看完的内容'
    },
    # 社交互动优先
    '社交互动优先': {
        'is_click': 0.5, 'is_like': 2.0, 'is_follow': 3.0, 'is_comment': 3.0,
        'is_forward': 3.0, 'long_view': 1.0, 'is_hate': -1.0, 'is_profile_enter': 0.5,
        'desc': '关注/评论/转发权重3x，强调社交传播价值'
    },
    # 仅正向信号（忽略点击和负反馈）
    '纯正向信号': {
        'is_click': 0, 'is_like': 1.0, 'is_follow': 2.0, 'is_comment': 2.0,
        'is_forward': 2.0, 'long_view': 2.0, 'is_hate': 0, 'is_profile_enter': 1.0,
        'desc': '不用点击和负反馈，只靠正向互动信号排序'
    },
}

K_VALUES = [10, 20, 30, 50]

def compute_relevance(y_true_dict):
    n = len(y_true_dict[TASKS[0]])
    rel = np.zeros(n)
    # relevance: 点击=1, 点赞=2, 关注=3, 评论=3, 转发=2, 有效播放=3, 负反馈=-2, 主页=1
    w = {'is_click': 1, 'is_like': 2, 'is_follow': 3, 'is_comment': 3,
         'is_forward': 2, 'long_view': 3, 'is_hate': -2, 'is_profile_enter': 1}
    for t, wt in w.items():
        rel += wt * y_true_dict[t]
    return np.maximum(rel, 0)

def compute_ndcg_at_k(y_true_dict, scores, k):
    n = len(scores)
    if n == 0:
        return 0.0
    relevance = compute_relevance(y_true_dict)
    ranking = np.argsort(-scores)
    dcg = sum(relevance[ranking[i]] / np.log2(i + 2) for i in range(min(k, n)))
    ideal = np.sort(relevance)[::-1]
    idcg = sum(ideal[i] / np.log2(i + 2) for i in range(min(k, n)))
    return dcg / (idcg + 1e-8)

def compute_gNDCG_at_k(y_true_dict, scores, user_ids, k):
    unique_users = np.unique(user_ids)
    ndcgs = []
    for u in unique_users:
        mask = user_ids == u
        if mask.sum() < k:
            continue
        ndcg = compute_ndcg_at_k(
            {t: y_true_dict[t][mask] for t in y_true_dict}, scores[mask], k)
        ndcgs.append(ndcg)
    return np.mean(ndcgs) if ndcgs else 0.0

def apply_fusion(predictions, weights):
    scores = np.zeros(predictions['n_samples'])
    for task, w in weights.items():
        if w == 0:
            continue
        p = np.clip(predictions['y_pred'][task], 1e-8, 1 - 1e-8)
        if w > 0:
            scores += w * np.log(p)
        else:
            # 负权重: 高概率 → 低分数
            scores += w * np.log(1 - p)
    return scores

models = {
    'SharedBottom': '/root/mtl4rec/results/predictions/SharedBottom_base_predictions.pkl',
    'MMoE_v3':     '/root/mtl4rec/results/predictions/MMoE_v3_predictions.pkl',
    'PLE_v7':      '/root/mtl4rec/results/predictions/PLE_v7_predictions.pkl',
}

# 加载所有预测
all_preds = {}
for name, path in models.items():
    with open(path, 'rb') as f:
        all_preds[name] = pickle.load(f)

# 计算所有策略 × 所有模型 × 所有K 的 NDCG
print("=" * 100)
print(f"{'策略':<16s} {'模型':<16s} {'NDCG@10':>8s} {'NDCG@20':>8s} {'NDCG@30':>8s} {'NDCG@50':>8s} {'gNDCG@10':>8s}")
print("=" * 100)

for sname, sweights in FUSION_STRATEGIES.items():
    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        pred = all_preds[mname]
        scores = apply_fusion(pred, {k: v for k, v in sweights.items() if k != 'desc'})
        
        ndcg10 = compute_ndcg_at_k(pred['y_true'], scores, 10)
        ndcg20 = compute_ndcg_at_k(pred['y_true'], scores, 20)
        ndcg30 = compute_ndcg_at_k(pred['y_true'], scores, 30)
        ndcg50 = compute_ndcg_at_k(pred['y_true'], scores, 50)
        gndcg10 = compute_gNDCG_at_k(pred['y_true'], scores, test_user_ids, 10)
        
        print(f"{sname:<16s} {mname:<16s} {ndcg10:8.4f} {ndcg20:8.4f} {ndcg30:8.4f} {ndcg50:8.4f} {gndcg10:8.4f}")
    print("-" * 100)

# 汇总：每个策略下 PLE_v7 是否最优
print()
print("=" * 80)
print(f"{'策略':<16s} {'最优模型':<16s} {'NDCG@10':>8s} {'说明'}")
print("=" * 80)
for sname, sweights in FUSION_STRATEGIES.items():
    best_model = None
    best_ndcg = -1
    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        pred = all_preds[mname]
        scores = apply_fusion(pred, {k: v for k, v in sweights.items() if k != 'desc'})
        ndcg10 = compute_ndcg_at_k(pred['y_true'], scores, 10)
        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            best_model = mname
    print(f"{sname:<16s} {best_model:<16s} {best_ndcg:8.4f}  {sweights.get('desc', '')}")
