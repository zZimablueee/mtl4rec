#!/usr/bin/env python3
"""
端到端 NDCG 评估 — 完整版
======================
流程：
1. 对每个模型推理 → 保存逐样本预测
2. 多目标融合 + 概率校准 → NDCG@K (K=10,20,30,50)
3. 相关性分析：click_pred vs relevance
4. 生成分析图表 + JSON 结果
5. 结果保存到本地工作区

用法：
  python3 ndcg_eval_full.py
"""
import os, sys, json, pickle, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_auc_score

# 配置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = '/root/mtl4rec'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'fuxictr'))
os.chdir(PROJECT_ROOT)

from fuxictr.utils import load_config
from fuxictr.pytorch.torch_utils import get_device
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.features import FeatureMap
import models as model_zoo

CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/KuaiRand1K')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results/analysis')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints/KuaiRand1K')
PRED_DIR = os.path.join(PROJECT_ROOT, 'results/predictions')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

TASK_NAMES = ['is_click', 'is_like', 'is_follow', 'is_comment',
              'is_forward', 'long_view', 'is_hate', 'is_profile_enter']
TASK_CN = {
    'is_click': '点击', 'is_like': '点赞', 'is_follow': '关注',
    'is_comment': '评论', 'is_forward': '转发', 'long_view': '有效播放',
    'is_hate': '负反馈', 'is_profile_enter': '主页进入'
}
K_VALUES = [10, 20, 30, 50]

# 模型配置映射: model_name → (config_filename, experiment_id)
MODEL_CONFIG_MAP = {
    'SharedBottom_base': ('model_config_v4', 'SharedBottom_KuaiRand1K'),
    'MMoE_v3':           ('model_config_v3', 'MMoE_KuaiRand1K'),
    'PLE_v7':            ('model_config_v7', 'PLE_KuaiRand1K'),
}

# relevance 权重
RELEVANCE_WEIGHTS = {
    'is_click': 1.0, 'is_like': 2.0, 'is_follow': 3.0,
    'is_comment': 3.0, 'is_forward': 2.0, 'long_view': 3.0,
    'is_hate': -2.0, 'is_profile_enter': 1.0,
}

# 融合策略
FUSION_STRATEGIES = {
    '仅点击': {'is_click': 1.0, 'is_like': 0, 'is_follow': 0, 'is_comment': 0,
               'is_forward': 0, 'long_view': 0, 'is_hate': 0, 'is_profile_enter': 0},
    '均衡融合': {'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 1.0,
                'is_forward': 1.0, 'long_view': 1.0, 'is_hate': 1.0, 'is_profile_enter': 1.0},
    '深度消费优先': {'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 0.5,
                  'is_forward': 0.5, 'long_view': 3.0, 'is_hate': 0.5, 'is_profile_enter': 1.5},
    '互动优先': {'is_click': 0.5, 'is_like': 2.0, 'is_follow': 2.0, 'is_comment': 2.0,
                'is_forward': 2.0, 'long_view': 1.0, 'is_hate': 0.5, 'is_profile_enter': 0.5},
    '降负反馈': {'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 1.0,
                'is_forward': 1.0, 'long_view': 1.0, 'is_hate': -1.0, 'is_profile_enter': 1.0},
}

# ============================================================
# Step 1: 推理
# ============================================================
def get_checkpoint_path(experiment_id):
    target = f"{experiment_id}.model"
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname == target:
            return os.path.join(CHECKPOINT_DIR, fname)
    return None

def run_inference(model_name):
    pred_file = os.path.join(PRED_DIR, f'{model_name}_predictions.pkl')
    if os.path.exists(pred_file):
        print(f"  [CACHE] {pred_file} 已存在")
        return pred_file
    
    config_name, experiment_id = MODEL_CONFIG_MAP.get(model_name, (None, None))
    if not config_name:
        print(f"  [SKIP] {model_name}: 无配置映射")
        return None
    
    ckpt_path = get_checkpoint_path(experiment_id)
    if not ckpt_path:
        print(f"  [SKIP] {model_name}: 找不到 checkpoint ({experiment_id}.model)")
        return None
    
    print(f"  [LOAD] {model_name}: config={config_name}, ckpt={os.path.basename(ckpt_path)}")
    
    params = load_config(CONFIG_DIR, experiment_id)
    params['gpu'] = 0
    
    feature_map_json = os.path.join(DATA_DIR, "feature_map.json")
    feature_map = FeatureMap(params['dataset_id'], DATA_DIR)
    feature_map.load(feature_map_json, params)
    feature_map.labels = TASK_NAMES
    feature_map.set_column_index()
    
    model_class = getattr(model_zoo, params['model'])
    device = get_device(params['gpu'])
    state_dict = torch.load(ckpt_path, map_location=device)
    
    for key in state_dict:
        if 'embedding_layer' in key and 'weight' in key:
            ckpt_embed_dim = state_dict[key].shape[-1]
            if ckpt_embed_dim != params.get('embedding_dim', 0):
                print(f"  [FIX] embedding_dim: {params.get('embedding_dim')} → {ckpt_embed_dim}")
                params['embedding_dim'] = ckpt_embed_dim
            break
    
    model = model_class(feature_map, **params)
    model = model.to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    
    print(f"  [INFER] 推理中...")
    y_pred_all = defaultdict(list)
    y_true_all = defaultdict(list)
    group_ids = []
    
    with torch.no_grad():
        for batch_data in test_gen:
            return_dict = model.forward(batch_data)
            batch_y_true = model.get_labels(batch_data)
            
            for i, task in enumerate(TASK_NAMES):
                y_pred_all[task].extend(
                    return_dict[f"{task}_pred"].data.cpu().numpy().reshape(-1).tolist())
                y_true_all[task].extend(
                    batch_y_true[i].data.cpu().numpy().reshape(-1).tolist())
            
            if feature_map.group_id is not None:
                group_ids.extend(model.get_group_id(batch_data).numpy().reshape(-1).tolist())
    
    predictions = {
        'model_name': model_name,
        'y_pred': {t: np.array(y_pred_all[t], dtype=np.float64) for t in TASK_NAMES},
        'y_true': {t: np.array(y_true_all[t], dtype=np.float64) for t in TASK_NAMES},
        'group_ids': np.array(group_ids) if group_ids else None,
        'n_samples': len(y_pred_all[TASK_NAMES[0]]),
    }
    
    with open(pred_file, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"  [SAVED] {pred_file} ({predictions['n_samples']} samples)")
    return pred_file

# ============================================================
# Step 2: AUC + gAUC
# ============================================================
def compute_auc_from_predictions(predictions):
    aucs = {}
    for task in TASK_NAMES:
        y_true = predictions['y_true'][task]
        y_pred = predictions['y_pred'][task]
        if len(np.unique(y_true)) < 2:
            aucs[task] = float('nan')
        else:
            aucs[task] = roc_auc_score(y_true, y_pred)
    aucs['avg_auc'] = np.nanmean(list(aucs.values()))
    return aucs

# ============================================================
# Step 3: 多目标融合 + NDCG
# ============================================================
def compute_relevance(y_true_dict):
    n = len(y_true_dict[TASK_NAMES[0]])
    relevance = np.zeros(n)
    for task, weight in RELEVANCE_WEIGHTS.items():
        relevance += weight * y_true_dict[task]
    return np.maximum(relevance, 0)

def compute_ndcg_at_k(y_true_dict, scores, k=10):
    n = len(scores)
    if n == 0:
        return 0.0
    relevance = compute_relevance(y_true_dict)
    ranking = np.argsort(-scores)
    dcg = sum(relevance[ranking[i]] / np.log2(i + 2) for i in range(min(k, n)))
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = sum(ideal_relevance[i] / np.log2(i + 2) for i in range(min(k, n)))
    return dcg / (idcg + 1e-8)

def compute_gNDCG_at_k(y_true_dict, scores, group_ids, k=10):
    if group_ids is None or len(group_ids) == 0:
        return compute_ndcg_at_k(y_true_dict, scores, k)
    unique_groups = np.unique(group_ids)
    ndcgs = []
    for g in unique_groups:
        mask = group_ids == g
        if mask.sum() < k:
            continue
        ndcg = compute_ndcg_at_k(
            {t: y_true_dict[t][mask] for t in y_true_dict}, scores[mask], k)
        ndcgs.append(ndcg)
    return np.mean(ndcgs) if ndcgs else 0.0

def apply_fusion(predictions, fusion_weights, calibration='none'):
    """多目标融合: Score = Σ w_k * log(p_k)"""
    scores = np.zeros(predictions['n_samples'])
    for task, weight in fusion_weights.items():
        if weight == 0:
            continue
        p = predictions['y_pred'][task]
        if calibration == 'temperature' and task == 'is_click':
            # Temperature scaling for is_click
            p_clipped = np.clip(p, 1e-8, 1 - 1e-8)
            logits = np.log(p_clipped / (1 - p_clipped))
            # Use T=2.0 as default (conservative)
            p = 1.0 / (1.0 + np.exp(-logits / 2.0))
        elif calibration == 'rank' and task == 'is_click':
            # Rank normalization
            temp = np.argsort(p)
            ranks = np.empty(len(p))
            ranks[temp] = np.arange(1, len(p) + 1) / len(p)
            p = ranks
        p = np.clip(p, 1e-8, 1 - 1e-8)
        scores += weight * np.log(p)
    return scores

# ============================================================
# Step 4: 相关性分析
# ============================================================
def correlation_analysis(predictions):
    """分析 click_pred 与 relevance 的相关性"""
    relevance = compute_relevance(predictions['y_true'])
    click_pred = predictions['y_pred']['is_click']
    
    corr = np.corrcoef(click_pred, relevance)[0, 1]
    
    # 按预测分位数分组
    n_bins = 10
    quantiles = np.percentile(click_pred, np.linspace(0, 100, n_bins + 1))
    bin_stats = []
    for i in range(n_bins):
        mask = (click_pred >= quantiles[i]) & (click_pred < quantiles[i + 1])
        if mask.sum() == 0:
            continue
        bin_stats.append({
            'bin': i,
            'n_samples': int(mask.sum()),
            'mean_pred': float(click_pred[mask].mean()),
            'mean_relevance': float(relevance[mask].mean()),
            'click_rate': float(predictions['y_true']['is_click'][mask].mean()),
        })
    
    return {'correlation': float(corr), 'bin_stats': bin_stats}

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['SharedBottom_base', 'MMoE_v3', 'PLE_v7'],
                        help='Models to evaluate')
    args = parser.parse_args()
    
    all_results = {}
    
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"处理模型: {model_name}")
        print(f"{'='*60}")
        
        # Step 1: 推理
        pred_file = run_inference(model_name)
        if not pred_file:
            continue
        
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
        
        # Step 2: AUC
        aucs = compute_auc_from_predictions(predictions)
        print(f"\n  --- AUC ---")
        for task in TASK_NAMES:
            print(f"    {task:20s}: AUC={aucs[task]:.4f}")
        print(f"    {'avg_auc':20s}: {aucs['avg_auc']:.4f}")
        
        # Step 4: 相关性分析
        corr = correlation_analysis(predictions)
        print(f"\n  --- 相关性 ---")
        print(f"    corr(click_pred, relevance): {corr['correlation']:.4f}")
        
        # Step 3: NDCG 评估
        model_results = {'auc': aucs, 'correlation': corr, 'ndcg': {}}
        
        for strategy_name, fusion_weights in FUSION_STRATEGIES.items():
            for calib in ['none', 'temperature', 'rank']:
                scores = apply_fusion(predictions, fusion_weights, calibration=calib)
                
                key = f"{strategy_name}_{calib}"
                ndcg_results = {}
                for k in K_VALUES:
                    ndcg = compute_ndcg_at_k(predictions['y_true'], scores, k)
                    gndcg = compute_gNDCG_at_k(predictions['y_true'], scores, predictions['group_ids'], k)
                    ndcg_results[f'NDCG@{k}'] = float(ndcg)
                    ndcg_results[f'gNDCG@{k}'] = float(gndcg)
                
                model_results['ndcg'][key] = ndcg_results
        
        all_results[model_name] = model_results
        
        # 打印 NDCG 摘要
        print(f"\n  --- NDCG@K (仅点击, 未校准) ---")
        for k in K_VALUES:
            key = f"仅点击_none"
            ndcg = model_results['ndcg'][key][f'NDCG@{k}']
            gndcg = model_results['ndcg'][key][f'gNDCG@{k}']
            print(f"    NDCG@{k:2d} = {ndcg:.4f}  |  gNDCG@{k:2d} = {gndcg:.4f}")
    
    # 保存结果
    result_path = os.path.join(OUTPUT_DIR, 'ndcg_results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存: {result_path}")
    
    # ============================================================
    # 生成对比图表
    # ============================================================
    print("\n生成图表...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    model_names = list(all_results.keys())
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    # 1. AUC 对比
    ax = axes[0, 0]
    x = np.arange(len(TASK_NAMES))
    width = 0.25
    for i, (name, color) in enumerate(zip(model_names, colors)):
        aucs = [all_results[name]['auc'][t] for t in TASK_NAMES]
        ax.bar(x + i * width, aucs, width, label=name, color=color, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([TASK_CN[t] for t in TASK_NAMES], rotation=45, ha='right')
    ax.set_ylabel('AUC')
    ax.set_title('各任务 AUC 对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. NDCG@K 对比 (仅点击, 未校准)
    ax = axes[0, 1]
    for name, color in zip(model_names, colors):
        ndcgs = [all_results[name]['ndcg']['仅点击_none'][f'NDCG@{k}'] for k in K_VALUES]
        ax.plot(K_VALUES, ndcgs, 'o-', label=name, color=color, linewidth=2)
    ax.set_xlabel('K')
    ax.set_ylabel('NDCG@K')
    ax.set_title('NDCG@K 对比 (仅点击, 无校准)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. gNDCG@K 对比
    ax = axes[0, 2]
    for name, color in zip(model_names, colors):
        gndcgs = [all_results[name]['ndcg']['仅点击_none'][f'gNDCG@{k}'] for k in K_VALUES]
        ax.plot(K_VALUES, gndcgs, 'o-', label=name, color=color, linewidth=2)
    ax.set_xlabel('K')
    ax.set_ylabel('gNDCG@K')
    ax.set_title('gNDCG@K 对比 (按用户分组)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. 校准效果对比 (PLE_v7)
    ax = axes[1, 0]
    if 'PLE_v7' in all_results:
        for calib_name, calib_label in [('none', '无校准'), ('temperature', '温度缩放'), ('rank', '排序归一化')]:
            key = f"仅点击_{calib_name}"
            ndcgs = [all_results['PLE_v7']['ndcg'][key][f'NDCG@{k}'] for k in K_VALUES]
            ax.plot(K_VALUES, ndcgs, 'o-', label=calib_label, linewidth=2)
        ax.set_xlabel('K')
        ax.set_ylabel('NDCG@K')
        ax.set_title('PLE_v7 校准效果对比')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # 5. 融合策略对比 (PLE_v7, NDCG@10)
    ax = axes[1, 1]
    if 'PLE_v7' in all_results:
        strategies = list(FUSION_STRATEGIES.keys())
        ndcg_vals = [all_results['PLE_v7']['ndcg'][f'{s}_none']['NDCG@10'] for s in strategies]
        bars = ax.barh(strategies, ndcg_vals, color='#4CAF50', alpha=0.8)
        for bar, val in zip(bars, ndcg_vals):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')
        ax.set_xlabel('NDCG@10')
        ax.set_title('PLE_v7 不同融合策略 NDCG@10')
        ax.grid(axis='x', alpha=0.3)
    
    # 6. 相关性 vs NDCG@10 散点图
    ax = axes[1, 2]
    for name, color in zip(model_names, colors):
        corr_val = all_results[name]['correlation']['correlation']
        ndcg_val = all_results[name]['ndcg']['仅点击_none']['NDCG@10']
        ax.scatter(corr_val, ndcg_val, s=200, color=color, label=name, edgecolors='black', linewidth=1)
        ax.annotate(name, (corr_val, ndcg_val), textcoords="offset points", xytext=(10, 10), fontsize=10)
    ax.set_xlabel('corr(click_pred, relevance)')
    ax.set_ylabel('NDCG@10')
    ax.set_title('预测校准 vs 排序质量')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'ndcg_analysis.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {chart_path}")
    
    # ============================================================
    # 打印最终对比表
    # ============================================================
    print("\n" + "=" * 80)
    print("=== 最终结果汇总 ===")
    print("=" * 80)
    
    # AUC 表
    print(f"\n{'模型':<20s} {'avg AUC':>8s}", end="")
    for t in TASK_NAMES:
        print(f" {TASK_CN[t]:>6s}", end="")
    print()
    print("-" * 100)
    for name in model_names:
        print(f"{name:<20s} {all_results[name]['auc']['avg_auc']:>8.4f}", end="")
        for t in TASK_NAMES:
            print(f" {all_results[name]['auc'][t]:>6.4f}", end="")
        print()
    
    # NDCG 表
    print(f"\n{'模型':<20s} {'corr':>6s}", end="")
    for k in K_VALUES:
        print(f" {'NDCG@'+str(k):>10s} {'gNDCG@'+str(k):>10s}", end="")
    print()
    print("-" * 120)
    for name in model_names:
        print(f"{name:<20s} {all_results[name]['correlation']['correlation']:>6.4f}", end="")
        for k in K_VALUES:
            ndcg = all_results[name]['ndcg']['仅点击_none'][f'NDCG@{k}']
            gndcg = all_results[name]['ndcg']['仅点击_none'][f'gNDCG@{k}']
            print(f" {ndcg:>10.4f} {gndcg:>10.4f}", end="")
        print()
    
    return all_results

if __name__ == '__main__':
    main()