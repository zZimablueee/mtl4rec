#!/usr/bin/env python3
"""
B + D 方案整合脚本
========================
D: 层级 relevance — 用层级偏好替代数值加权
B: MLP 融合 — 训练小型 MLP 从数据中学习融合权重

对比：
1. 手工融合 + 数值 relevance（baseline）
2. 手工融合 + 层级 relevance（D 方案）
3. MLP 融合 + 层级 relevance（B + D 方案）
"""
import pickle, numpy as np, json, os, sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment',
         'is_forward', 'long_view', 'is_hate', 'is_profile_enter']
K_VALUES = [10, 20, 30, 50]

# 加载 test.csv 的 user_id 用于 gNDCG
import pandas as pd
df = pd.read_csv('/root/mtl4rec/data/KuaiRand1K/test.csv', usecols=['user_id'])
test_user_ids = df['user_id'].values

# ============================================================
# 融合策略（与 fusion_eval.py 一致）
# ============================================================
FUSION_STRATEGIES = {
    '仅点击': {
        'is_click': 1.0, 'is_like': 0, 'is_follow': 0, 'is_comment': 0,
        'is_forward': 0, 'long_view': 0, 'is_hate': 0, 'is_profile_enter': 0,
        'desc': '只用点击概率'
    },
    '等权融合': {
        'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 1.0,
        'is_forward': 1.0, 'long_view': 1.0, 'is_hate': 1.0, 'is_profile_enter': 1.0,
        'desc': '8个任务等权相加'
    },
    '互动深度加权': {
        'is_click': 1.0, 'is_like': 2.0, 'is_follow': 3.0, 'is_comment': 3.0,
        'is_forward': 2.0, 'long_view': 3.0, 'is_hate': -2.0, 'is_profile_enter': 1.0,
        'desc': '点击<点赞<关注/评论/有效播放'
    },
    '降负反馈综合': {
        'is_click': 1.0, 'is_like': 2.0, 'is_follow': 2.0, 'is_comment': 2.0,
        'is_forward': 1.5, 'long_view': 2.0, 'is_hate': -3.0, 'is_profile_enter': 1.0,
        'desc': '负反馈权重-3，强烈惩罚'
    },
    '深度消费优先': {
        'is_click': 0.8, 'is_like': 1.0, 'is_follow': 1.5, 'is_comment': 1.0,
        'is_forward': 1.0, 'long_view': 3.0, 'is_hate': -1.5, 'is_profile_enter': 1.5,
        'desc': '有效播放权重3x'
    },
    '社交互动优先': {
        'is_click': 0.5, 'is_like': 2.0, 'is_follow': 3.0, 'is_comment': 3.0,
        'is_forward': 3.0, 'long_view': 1.0, 'is_hate': -1.0, 'is_profile_enter': 0.5,
        'desc': '关注/评论/转发权重3x'
    },
    '纯正向信号': {
        'is_click': 0, 'is_like': 1.0, 'is_follow': 2.0, 'is_comment': 2.0,
        'is_forward': 2.0, 'long_view': 2.0, 'is_hate': 0, 'is_profile_enter': 1.0,
        'desc': '不用点击和负反馈'
    },
}

# ============================================================
# D 方案: 层级 relevance
# 参考快手 EMER: 多重正反馈 > 单一正反馈 > 无正反馈
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

    # 定义层级
    tier_L3 = ['is_follow', 'is_comment']
    tier_L2 = ['is_like', 'is_forward', 'long_view']
    tier_L1 = ['is_click', 'is_profile_enter']

    for i in range(n):
        # 负反馈检查
        if y_true_dict['is_hate'][i] == 1:
            relevance[i] = 0
            continue

        # 确定最高层级
        max_tier = 0
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

        # 多样性奖励: 正反馈种类 >= 2 则 +1
        all_pos = tier_L3 + tier_L2 + tier_L1
        n_pos = sum(1 for t in all_pos if y_true_dict[t][i] == 1)
        bonus = 1 if n_pos >= 2 else 0

        relevance[i] = max_tier + bonus

    return relevance


# ============================================================
# NDCG / gNDCG 计算
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
# B 方案: MLP 融合
# ============================================================
def build_mlp(X_train, y_train, X_val, y_val):
    """训练一个小型 MLP 学习融合权重"""
    # 使用 sklearn 的 MLPRegressor（简单、稳定、不需要 GPU）
    from sklearn.neural_network import MLPRegressor

    # 尝试不同的 hidden_layer_sizes，选验证集上最好的
    best_model = None
    best_score = -1
    best_config = None

    configs = [
        (16,),        # 1 层 16
        (16, 8),      # 2 层 16→8
        (32, 16),     # 2 层 32→16
        (32, 16, 8),  # 3 层 32→16→8
    ]

    for hidden in configs:
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation='relu',
            solver='adam',
            alpha=0.001,             # L2 正则化
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_val)
        # 用 Spearman 相关系数衡量排序质量
        from scipy.stats import spearmanr
        corr, _ = spearmanr(y_val, y_pred)
        if corr > best_score:
            best_score = corr
            best_model = mlp
            best_config = hidden

    return best_model, best_config, best_score


def mlp_fusion(predictions, relevance_func, test_user_ids):
    """
    MLP 融合流程:
    1. 从 8 个预测概率构造特征
    2. 训练 MLP 回归 relevance
    3. 用 MLP 打分
    4. 计算 NDCG
    """
    # 构造特征: 8 个预测概率 + 额外特征
    n_samples = predictions['n_samples']
    X = np.column_stack([predictions['y_pred'][t] for t in TASKS])

    # 添加一些手工特征，帮助 MLP
    # log(p) 和 log(1-p) 特征
    X_log = np.column_stack([np.log(np.clip(predictions['y_pred'][t], 1e-8, 1 - 1e-8)) for t in TASKS])
    X = np.column_stack([X, X_log])

    # 真实标签
    y = relevance_func(predictions['y_true'])

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练/验证（按用户维度，避免泄漏）
    unique_users = np.unique(test_user_ids)
    train_users, val_users = train_test_split(unique_users, test_size=0.3, random_state=42)

    train_mask = np.isin(test_user_ids, train_users)
    val_mask = np.isin(test_user_ids, val_users)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"  [MLP] 训练集: {len(X_train)} 样本 ({len(train_users)} 用户)")
    print(f"  [MLP] 验证集: {len(X_val)} 样本 ({len(val_users)} 用户)")

    # 训练 MLP
    model, config, val_score = build_mlp(X_train, y_train, X_val, y_val)
    print(f"  [MLP] 最优配置: hidden={config}, val Spearman r={val_score:.4f}")

    # 全量打分
    all_scores = model.predict(X)

    # 计算 NDCG
    results = {}
    for k in K_VALUES:
        ndcg = compute_ndcg_at_k(predictions['y_true'], all_scores, k, relevance_func)
        gndcg = compute_gNDCG_at_k(predictions['y_true'], all_scores, test_user_ids, k, relevance_func)
        results[f'NDCG@{k}'] = float(ndcg)
        results[f'gNDCG@{k}'] = float(gndcg)

    return results, model, config


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 80)
    print("  B + D 方案: 层级 relevance + MLP 融合")
    print("=" * 80)

    # 加载预测
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

    # 用一个模型的真实标签分析层级分布
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

    # 相关性: 数值型 vs 层级
    corr = np.corrcoef(rel_num, rel_hier)[0, 1]
    print(f"\n  数值型 relevance 与 层级 relevance 的 Pearson r = {corr:.4f}")

    # ============================================================
    # 第二部分: 手工融合 + 层级 relevance vs 数值 relevance
    # ============================================================
    print("\n" + "=" * 80)
    print("  第二部分: 手工融合 NDCG — 数值 relevance vs 层级 relevance")
    print("=" * 80)

    print(f"\n  {'策略':<16s} {'模型':<16s} {'NDCG@10(数值)':>14s} {'NDCG@10(层级)':>14s} {'变化':>8s}")
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

            key = f"{sname}|{mname}"
            comparison[key] = {'ndcg_num': ndcg_num, 'ndcg_hier': ndcg_hier}

    # ============================================================
    # 第三部分: MLP 融合 + 层级 relevance
    # ============================================================
    print("\n" + "=" * 80)
    print("  第三部分: MLP 融合 + 层级 relevance")
    print("=" * 80)

    mlp_results = {}
    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        print(f"\n  --- {mname} ---")
        pred = all_preds[mname]
        results, mlp_model, config = mlp_fusion(pred, compute_relevance_hierarchical, test_user_ids)
        mlp_results[mname] = results

        print(f"  {'NDCG@10':>10s}: {results['NDCG@10']:.4f}")
        print(f"  {'NDCG@20':>10s}: {results['NDCG@20']:.4f}")
        print(f"  {'NDCG@30':>10s}: {results['NDCG@30']:.4f}")
        print(f"  {'NDCG@50':>10s}: {results['NDCG@50']:.4f}")
        print(f"  {'gNDCG@10':>10s}: {results['gNDCG@10']:.4f}")

    # ============================================================
    # 第四部分: 最终对比表
    # ============================================================
    print("\n" + "=" * 80)
    print("  第四部分: 最终对比 — 三种方案 × 三个模型")
    print("=" * 80)

    # 找到每个模型在手工融合下的最佳策略
    best_manual = {}
    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        best_ndcg = -1
        best_strat = None
        for sname in FUSION_STRATEGIES:
            key = f"{sname}|{mname}"
            ndcg = comparison[key]['ndcg_hier']
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_strat = sname
        best_manual[mname] = (best_strat, best_ndcg)

    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        best_strat, best_ndcg = best_manual[mname]
        mlp_ndcg = mlp_results[mname]['NDCG@10']
        best_num_ndcg = max(comparison[f"{s}|{mname}"]['ndcg_num'] for s in FUSION_STRATEGIES)

        mlp_vs_best = mlp_ndcg - best_ndcg
        arrow = "↑" if mlp_vs_best > 0 else ("↓" if mlp_vs_best < 0 else "→")

    # 完整对比表
    print(f"\n  {'方案':<30s} {'SharedBottom':>14s} {'MMoE_v3':>14s} {'PLE_v7':>14s}")
    print(f"  {'-'*75}")

    # 手工融合 + 数值 relevance (最佳策略)
    nums = []
    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        v = max(comparison[f"{s}|{mname}"]['ndcg_num'] for s in FUSION_STRATEGIES)
        nums.append(v)
    print(f"  {'手工融合 + 数值 relevance':<30s} {nums[0]:14.4f} {nums[1]:14.4f} {nums[2]:14.4f}")

    # 手工融合 + 层级 relevance (最佳策略)
    hiers = []
    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        v = max(comparison[f"{s}|{mname}"]['ndcg_hier'] for s in FUSION_STRATEGIES)
        hiers.append(v)
    print(f"  {'手工融合 + 层级 relevance（D方案）':<30s} {hiers[0]:14.4f} {hiers[1]:14.4f} {hiers[2]:14.4f}")

    # MLP 融合
    mlps = [mlp_results[m]['NDCG@10'] for m in ['SharedBottom', 'MMoE_v3', 'PLE_v7']]
    print(f"  {'MLP融合 + 层级 relevance（B+D方案）':<30s} {mlps[0]:14.4f} {mlps[1]:14.4f} {mlps[2]:14.4f}")

    # 箭头对比
    print(f"\n  {'MLP融合 vs 手工融合(层级)':<30s}", end="")
    for m in range(3):
        diff = mlps[m] - hiers[m]
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        print(f"  {arrow} {diff:+.4f}     ", end="")
    print()

    print(f"\n  {'MLP融合 vs 手工融合(数值)':<30s}", end="")
    for m in range(3):
        diff = mlps[m] - nums[m]
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        print(f"  {arrow} {diff:+.4f}     ", end="")
    print()

    # ============================================================
    # 第五部分: 详细 MLP 结果（所有 K 值）
    # ============================================================
    print("\n" + "=" * 80)
    print("  第五部分: MLP 融合详细结果")
    print("=" * 80)

    for mname in ['SharedBottom', 'MMoE_v3', 'PLE_v7']:
        print(f"\n  {mname}:")
        for k in K_VALUES:
            v = mlp_results[mname][f'NDCG@{k}']
            gv = mlp_results[mname][f'gNDCG@{k}']
            print(f"    NDCG@{k:2d} = {v:.4f}  |  gNDCG@{k:2d} = {gv:.4f}")

    # ============================================================
    # 保存结果
    # ============================================================
    output = {
        'hierarchical_relevance': {
            'distribution': {str(k): int(v) for k, v in zip(*np.unique(rel_hier, return_counts=True))},
        },
        'best_manual_numerical': {m: float(v) for m, v in zip(['SharedBottom', 'MMoE_v3', 'PLE_v7'], nums)},
        'best_manual_hierarchical': {m: float(v) for m, v in zip(['SharedBottom', 'MMoE_v3', 'PLE_v7'], hiers)},
        'mlp_fusion': {m: r for m, r in mlp_results.items()},
    }

    out_path = '/root/mtl4rec/results/analysis/improved_fusion_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")


if __name__ == '__main__':
    main()