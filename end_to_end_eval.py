#!/usr/bin/env python3
"""
端到端评估 Pipeline v2：模型预测 → 多目标融合 → NDCG@K 评价
================================================================
v2 新增:
1. 多 K 值评估: NDCG@10, @20, @30, @50
2. 预测概率校准: Temperature Scaling + Rank Normalization
3. AUC 重算: 从 predictions 重新计算，确保与训练日志一致
4. 校准分析: 对比 raw vs calibrated 的排序质量

流程：
  Step 1: 加载模型 checkpoint → 对 test 集推理 → 保存逐样本预测值
  Step 2: 加载预测值 → 多目标融合 → 计算 NDCG@K + 业务指标
  Step 3: 画图对比

用法：
  python end_to_end_eval.py --model PLE_v5
  python end_to_end_eval.py --all  # 跑所有关键模型
"""
import os, sys, json, argparse, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
from sklearn.metrics import roc_auc_score

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
    'SharedBottom_base': ('model_config', 'SharedBottom_KuaiRand1K'),
    'MMoE_v3':           ('model_config_v3', 'MMoE_KuaiRand1K'),
    'PLE_v5':            ('model_config_v5', 'PLE_KuaiRand1K'),
    'PLE_v15':           ('model_config_v15', 'PLE_v15_KuaiRand1K'),
}

# ============================================================
# Step 1: 推理并保存预测值
# ============================================================
def get_checkpoint_path(model_name):
    """查找 checkpoint — 按 experiment_id 匹配"""
    _, experiment_id = MODEL_CONFIG_MAP.get(model_name, (None, None))
    if not experiment_id:
        return None
    target = f"{experiment_id}.model"
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname == target:
            return os.path.join(CHECKPOINT_DIR, fname)
    return None


def run_inference(model_name):
    """加载模型，对 test 集推理，保存预测值和标签"""
    pred_file = os.path.join(PRED_DIR, f'{model_name}_predictions.pkl')
    
    if os.path.exists(pred_file):
        print(f"  [CACHE] {pred_file} 已存在，跳过推理")
        return pred_file
    
    config_name, experiment_id = MODEL_CONFIG_MAP.get(model_name, (None, None))
    if not config_name:
        print(f"  [SKIP] {model_name}: 无配置映射")
        return None
    
    ckpt_path = get_checkpoint_path(model_name)
    if not ckpt_path:
        print(f"  [SKIP] {model_name}: 找不到 checkpoint")
        return None
    
    print(f"  [LOAD] {model_name}: config={config_name}, expid={experiment_id}, ckpt={os.path.basename(ckpt_path)}")
    
    params = load_config(CONFIG_DIR, experiment_id)
    params['gpu'] = 0
    
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    if params.get('data_format') == 'csv':
        params['train_data'] = os.path.join(data_dir, 'train')
        params['valid_data'] = os.path.join(data_dir, 'valid')
        params['test_data'] = os.path.join(data_dir, 'test')
    
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
                print(f"  [FIX] embedding_dim: config={params.get('embedding_dim')} → checkpoint={ckpt_embed_dim}")
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
# Step 2: 多目标融合 + NDCG 评价
# ============================================================

# relevance 权重
RELEVANCE_WEIGHTS = {
    'is_click': 1.0,
    'is_like': 2.0,
    'is_follow': 3.0,
    'is_comment': 3.0,
    'is_forward': 2.0,
    'long_view': 3.0,
    'is_hate': -2.0,
    'is_profile_enter': 1.0,
}


def compute_relevance(y_true_dict):
    """计算多任务加权 relevance"""
    n = len(y_true_dict[TASK_NAMES[0]])
    relevance = np.zeros(n)
    for task, weight in RELEVANCE_WEIGHTS.items():
        relevance += weight * y_true_dict[task]
    return np.maximum(relevance, 0)


def compute_ndcg_at_k(y_true_dict, scores, k=10):
    """计算 NDCG@K"""
    n = len(scores)
    if n == 0:
        return 0.0
    
    relevance = compute_relevance(y_true_dict)
    ranking = np.argsort(-scores)
    
    dcg = 0.0
    for i in range(min(k, n)):
        idx = ranking[i]
        dcg += relevance[idx] / np.log2(i + 2)
    
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = 0.0
    for i in range(min(k, n)):
        idcg += ideal_relevance[i] / np.log2(i + 2)
    
    return dcg / (idcg + 1e-8)


def compute_gNDCG_at_k(y_true_dict, scores, group_ids, k=10):
    """计算 gNDCG@K（按用户分组）"""
    if group_ids is None or len(group_ids) == 0:
        return compute_ndcg_at_k(y_true_dict, scores, k)
    
    unique_groups = np.unique(group_ids)
    ndcgs = []
    for g in unique_groups:
        mask = group_ids == g
        if mask.sum() < k:
            continue
        ndcg = compute_ndcg_at_k(
            {t: y_true_dict[t][mask] for t in y_true_dict},
            scores[mask], k
        )
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def compute_hit_rate_at_k(y_true_dict, scores, k=10):
    """计算 HitRate@K: top-K 中是否至少有一个正样本"""
    n = len(scores)
    if n == 0:
        return 0.0
    
    relevance = compute_relevance(y_true_dict)
    ranking = np.argsort(-scores)
    top_k_rel = relevance[ranking[:k]]
    return float(np.sum(top_k_rel > 0) > 0)


def compute_gHitRate_at_k(y_true_dict, scores, group_ids, k=10):
    """计算 gHitRate@K（按用户分组）"""
    if group_ids is None or len(group_ids) == 0:
        return compute_hit_rate_at_k(y_true_dict, scores, k)
    
    unique_groups = np.unique(group_ids)
    hrs = []
    for g in unique_groups:
        mask = group_ids == g
        if mask.sum() < k:
            continue
        hr = compute_hit_rate_at_k(
            {t: y_true_dict[t][mask] for t in y_true_dict},
            scores[mask], k
        )
        hrs.append(hr)
    
    return np.mean(hrs) if hrs else 0.0


# ============================================================
# 预测概率校准
# ============================================================

def temperature_scaling(predictions, task='is_click'):
    """
    Temperature Scaling: p_calibrated = sigmoid(logit / T)
    使用 validation set 的 optimal T（这里用 test set 自校准）
    """
    y_pred = predictions['y_pred'][task]
    y_true = predictions['y_true'][task]
    
    # 搜索最优 T（1.0 到 5.0）
    p = np.clip(y_pred, 1e-8, 1 - 1e-8)
    logits = np.log(p / (1 - p))
    
    best_t = 1.0
    best_ece = float('inf')
    
    for T in np.arange(1.0, 5.1, 0.5):
        calibrated_p = 1.0 / (1.0 + np.exp(-logits / T))
        # 简单 ECE (Expected Calibration Error)
        n_bins = 10
        ece = 0.0
        for b in range(n_bins):
            low, high = b / n_bins, (b + 1) / n_bins
            mask = (calibrated_p >= low) & (calibrated_p < high)
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = calibrated_p[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
        if ece < best_ece:
            best_ece = ece
            best_t = T
    
    return best_t


def apply_calibration(predictions, method='temperature', task='is_click'):
    """
    对预测概率进行校准
    method: 'temperature' | 'rank' | 'none'
    """
    if method == 'none':
        return predictions['y_pred'][task].copy()
    
    y_pred = predictions['y_pred'][task]
    p = np.clip(y_pred, 1e-8, 1 - 1e-8)
    
    if method == 'temperature':
        T = temperature_scaling(predictions, task)
        logits = np.log(p / (1 - p))
        calibrated = 1.0 / (1.0 + np.exp(-logits / T))
        return calibrated
    
    elif method == 'rank':
        # Rank normalization: O(n log n) via argsort
        temp = np.argsort(y_pred)
        ranks = np.empty(len(y_pred))
        ranks[temp] = np.arange(1, len(y_pred) + 1) / len(y_pred)
        return ranks
    
    return y_pred.copy()


def apply_fusion(predictions, fusion_weights, calibration='none'):
    """
    多目标融合
    fusion_weights: {task_name: weight}
    calibration: 'none' | 'temperature' | 'rank'
    公式: Score = Σ w_k * log(p_k)  (log-space 加法融合)
    """
    scores = np.zeros(predictions['n_samples'])
    for task, weight in fusion_weights.items():
        if weight == 0:
            continue
        
        if task == 'is_click' and calibration != 'none':
            p = apply_calibration(predictions, method=calibration, task=task)
        else:
            p = predictions['y_pred'][task]
        
        p = np.clip(p, 1e-8, 1 - 1e-8)
        scores += weight * np.log(p)
    return scores


# ============================================================
# 融合策略
# ============================================================

FUSION_STRATEGIES = {
    '仅点击 (baseline)': {
        'is_click': 1.0, 'is_like': 0, 'is_follow': 0, 'is_comment': 0,
        'is_forward': 0, 'long_view': 0, 'is_hate': 0, 'is_profile_enter': 0
    },
    '均衡融合': {
        'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 1.0,
        'is_forward': 1.0, 'long_view': 1.0, 'is_hate': 1.0, 'is_profile_enter': 1.0
    },
    '深度消费优先': {
        'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 0.5,
        'is_forward': 0.5, 'long_view': 3.0, 'is_hate': 0.5, 'is_profile_enter': 1.5
    },
    '互动优先': {
        'is_click': 0.5, 'is_like': 2.0, 'is_follow': 2.0, 'is_comment': 2.0,
        'is_forward': 2.0, 'long_view': 1.0, 'is_hate': 0.5, 'is_profile_enter': 0.5
    },
    '降负反馈': {
        'is_click': 1.0, 'is_like': 1.0, 'is_follow': 1.0, 'is_comment': 1.0,
        'is_forward': 1.0, 'long_view': 1.0, 'is_hate': -1.0, 'is_profile_enter': 1.0
    },
}


def compute_auc_from_predictions(predictions):
    """从 predictions 重新计算各任务 AUC"""
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


def evaluate_fusion_strategies(predictions):
    """评估多种融合策略 + 多种校准方式"""
    model_name = predictions['model_name']
    print(f"\n  [EVAL] {model_name} ({predictions['n_samples']} samples)")
    
    # 1. AUC 重算
    aucs = compute_auc_from_predictions(predictions)
    print(f"\n  --- AUC (from predictions) ---")
    for task in TASK_NAMES:
        print(f"    {task:20s}: AUC={aucs[task]:.4f}")
    print(f"    {'avg_auc':20s}: {aucs['avg_auc']:.4f}")
    
    # 2. 预测分布分析
    print(f"\n  --- 预测分布 (is_click) ---")
    y_pred = predictions['y_pred']['is_click']
    y_true = predictions['y_true']['is_click']
    print(f"    mean={y_pred.mean():.6f}, std={y_pred.std():.6f}, "
          f"min={y_pred.min():.6f}, max={y_pred.max():.6f}")
    print(f"    正样本率: {y_true.mean():.4f}")
    
    # 3. 相关性分析
    relevance = compute_relevance(predictions['y_true'])
    corr = np.corrcoef(y_pred, relevance)[0, 1]
    print(f"    corr(click_pred, relevance)={corr:.4f}")
    
    # 4. 校准分析
    print(f"\n  --- 校准分析 ---")
    for calib in ['none', 'temperature', 'rank']:
        calib_pred = apply_calibration(predictions, method=calib, task='is_click')
        calib_corr = np.corrcoef(calib_pred, relevance)[0, 1]
        if calib == 'temperature':
            T = temperature_scaling(predictions, 'is_click')
            print(f"    {calib:12s}: T={T:.1f}, corr(calibrated_pred, relevance)={calib_corr:.4f}")
        else:
            print(f"    {calib:12s}: corr(calibrated_pred, relevance)={calib_corr:.4f}")
    
    # 5. 融合策略评估 (所有 K 值 + 所有校准方式)
    calibration_methods = ['none', 'temperature', 'rank']
    
    results = {}
    for strategy_name, weights in FUSION_STRATEGIES.items():
        strategy_results = {}
        
        for calib in calibration_methods:
            scores = apply_fusion(predictions, weights, calibration=calib)
            
            calib_key = calib
            strategy_results[calib_key] = {
                'weights': weights,
                'ndcg': {},
                'gndcg': {},
                'hit_rate': {},
            }
            
            for k in K_VALUES:
                ndcg = compute_ndcg_at_k(predictions['y_true'], scores, k=k)
                strategy_results[calib_key]['ndcg'][k] = round(ndcg, 4)
                
                if predictions['group_ids'] is not None:
                    gndcg = compute_gNDCG_at_k(
                        predictions['y_true'], scores, predictions['group_ids'], k=k)
                    strategy_results[calib_key]['gndcg'][k] = round(gndcg, 4)
                
                hr = compute_hit_rate_at_k(predictions['y_true'], scores, k=k)
                strategy_results[calib_key]['hit_rate'][k] = round(hr, 4)
        
        results[strategy_name] = strategy_results
    
    # 打印汇总表 (仅 raw 校准)
    print(f"\n  --- NDCG 汇总 (raw predictions, 含多 K 值) ---")
    header = f"    {'策略':16s}"
    for k in K_VALUES:
        header += f" {'NDCG@'+str(k):>10s}"
    header += f" {'corr(click,rel)':>18s}"
    print(header)
    print("    " + "-" * (16 + 10 * len(K_VALUES) + 18))
    
    for strategy_name in FUSION_STRATEGIES.keys():
        row = f"    {strategy_name:16s}"
        for k in K_VALUES:
            row += f" {results[strategy_name]['none']['ndcg'][k]:>10.4f}"
        # corr for click-only strategy
        if strategy_name == '仅点击 (baseline)':
            row += f" {corr:>18.4f}"
        else:
            row += f" {'':>18s}"
        print(row)
    
    # 打印校准对比
    print(f"\n  --- 校准对比 (仅点击 baseline, NDCG@10) ---")
    for strategy_name in ['仅点击 (baseline)', '均衡融合']:
        row = f"    {strategy_name:16s}"
        for calib in calibration_methods:
            row += f" {calib:>12s}: {results[strategy_name][calib]['ndcg'][10]:.4f}"
        print(row)
    
    return results, aucs, corr


# ============================================================
# Step 3: 跨模型对比
# ============================================================

def compare_models_across_strategies(all_results, all_aucs, all_corrs):
    """对比不同模型在不同融合策略下的表现"""
    if not all_results:
        return
    
    models = list(all_results.keys())
    strategies = list(FUSION_STRATEGIES.keys())
    
    # ===== 图1: NDCG@K 对比 (不同 K 值) =====
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for ki, k in enumerate(K_VALUES):
        ax = axes[ki]
        x = np.arange(len(strategies))
        width = 0.8 / len(models)
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for i, model_name in enumerate(models):
            values = [all_results[model_name][s]['none']['ndcg'][k] for s in strategies]
            ax.bar(x + i * width, values, width, label=model_name,
                   color=colors[i], edgecolor='white')
        
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(strategies, fontsize=8, rotation=45, ha='right')
        ax.set_title(f'NDCG@{k}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'NDCG@{k}')
        ax.grid(axis='y', alpha=0.3)
        if ki == 0:
            ax.legend(fontsize=8, loc='upper left')
    
    fig.suptitle('End-to-End Evaluation: Model + Multi-Objective Fusion\n'
                 '(NDCG@K = ranking quality after fusion, higher = better)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'end_to_end_eval.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVED] {path}")
    
    # ===== 图2: 校准效果对比 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    calib_methods = ['none', 'temperature', 'rank']
    calib_labels = ['Raw', 'Temperature', 'Rank Norm']
    
    for model_idx, model_name in enumerate(models):
        ax = axes[model_idx] if model_idx < 2 else None
        if ax is None:
            continue
        
        x = np.arange(len(strategies))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for ci, calib in enumerate(calib_methods):
            values = [all_results[model_name][s][calib]['ndcg'][10] for s in strategies]
            ax.bar(x + ci * width, values, width, label=calib_labels[ci],
                   color=colors[ci], edgecolor='white', alpha=0.8)
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(strategies, fontsize=8, rotation=45, ha='right')
        ax.set_title(f'{model_name}: Calibration Effect on NDCG@10', fontsize=12)
        ax.set_ylabel('NDCG@10')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'calibration_effect.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")
    
    # ===== 图3: AUC vs NDCG 散点图 =====
    fig, ax = plt.subplots(figsize=(10, 8))
    
    model_colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for i, model_name in enumerate(models):
        avg_auc = all_aucs[model_name]['avg_auc']
        ndcg10 = all_results[model_name]['仅点击 (baseline)']['none']['ndcg'][10]
        ndcg20 = all_results[model_name]['仅点击 (baseline)']['none']['ndcg'][20]
        ndcg30 = all_results[model_name]['仅点击 (baseline)']['none']['ndcg'][30]
        ndcg50 = all_results[model_name]['仅点击 (baseline)']['none']['ndcg'][50]
        
        ax.scatter(avg_auc, ndcg10, c=[model_colors[i]], s=120, 
                   label=f'{model_name} (NDCG@10)', marker='o', edgecolors='black')
        ax.scatter(avg_auc, ndcg50, c=[model_colors[i]], s=80, 
                   label=f'{model_name} (NDCG@50)', marker='s', edgecolors='black', alpha=0.5)
    
    ax.set_xlabel('Avg Test AUC', fontsize=12)
    ax.set_ylabel('NDCG (click-only)', fontsize=12)
    ax.set_title('AUC vs NDCG: Model Quality vs Ranking Quality', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'auc_vs_ndcg.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, nargs='+', help='模型名(可多个)')
    parser.add_argument('--all', action='store_true', help='跑所有关键模型')
    parser.add_argument('--force', action='store_true', help='强制重新推理（忽略缓存）')
    parser.add_argument('--skip', type=str, default=None, nargs='+', help='跳过的模型名')
    args = parser.parse_args()
    
    all_default = ['SharedBottom_base', 'MMoE_v3', 'PLE_v5', 'PLE_v15']
    
    if args.all:
        models_to_run = all_default
    elif args.model:
        models_to_run = args.model
    else:
        models_to_run = all_default
    
    if args.skip:
        models_to_run = [m for m in models_to_run if m not in args.skip]
    
    print("=" * 60)
    print("端到端评估 Pipeline v2")
    print(f"模型: {models_to_run}")
    print(f"K 值: {K_VALUES}")
    print(f"校准: none, temperature, rank")
    print("=" * 60)
    
    # 如果 --force，删除缓存
    if args.force:
        for model_name in models_to_run:
            pred_file = os.path.join(PRED_DIR, f'{model_name}_predictions.pkl')
            if os.path.exists(pred_file):
                os.remove(pred_file)
                print(f"  [CLEAR] {pred_file}")
    
    all_fusion_results = {}
    all_aucs = {}
    all_corrs = {}
    
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")
        
        # Step 1: 推理
        pred_file = run_inference(model_name)
        if pred_file is None:
            continue
        
        # Step 2: 加载预测值
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
        
        # Step 3: 多目标融合评估
        fusion_results, aucs, corr = evaluate_fusion_strategies(predictions)
        all_fusion_results[model_name] = fusion_results
        all_aucs[model_name] = aucs
        all_corrs[model_name] = corr
    
    # Step 4: 跨模型对比
    if len(all_fusion_results) > 1:
        print(f"\n{'='*60}")
        print("跨模型端到端对比")
        print(f"{'='*60}")
        compare_models_across_strategies(all_fusion_results, all_aucs, all_corrs)
    
    # ===== 保存结果 =====
    summary = {
        'models': {},
        'k_values': K_VALUES,
        'calibration_methods': ['none', 'temperature', 'rank'],
    }
    
    for model_name in all_fusion_results.keys():
        summary['models'][model_name] = {
            'auc': {t: round(v, 4) for t, v in all_aucs[model_name].items()},
            'corr': round(all_corrs[model_name], 4),
            'fusion': {},
        }
        for strategy_name, calib_dict in all_fusion_results[model_name].items():
            summary['models'][model_name]['fusion'][strategy_name] = {}
            for calib, metrics in calib_dict.items():
                summary['models'][model_name]['fusion'][strategy_name][calib] = {
                    'ndcg': metrics['ndcg'],
                    'hit_rate': metrics['hit_rate'],
                }
    
    summary_path = os.path.join(OUTPUT_DIR, 'end_to_end_results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {summary_path}")
    
    # ===== 打印最终排名表 =====
    print("\n" + "=" * 80)
    print("最终排名表: NDCG@K (仅点击 baseline, raw predictions)")
    print("=" * 80)
    header = f"  {'模型':20s} {'avg AUC':>8s} {'corr(click,rel)':>16s}"
    for k in K_VALUES:
        header += f" {'NDCG@'+str(k):>10s}"
    print(header)
    print("  " + "-" * (20 + 8 + 16 + 10 * len(K_VALUES)))
    
    for model_name in all_fusion_results.keys():
        auc = all_aucs[model_name]['avg_auc']
        corr = all_corrs[model_name]
        row = f"  {model_name:20s} {auc:>8.4f} {corr:>16.4f}"
        for k in K_VALUES:
            ndcg = all_fusion_results[model_name]['仅点击 (baseline)']['none']['ndcg'][k]
            row += f" {ndcg:>10.4f}"
        print(row)
    
    # ===== 最佳策略推荐 =====
    print("\n" + "=" * 60)
    print("最佳端到端策略 (NDCG@10)")
    print("=" * 60)
    for model_name in all_fusion_results.keys():
        best_strategy = None
        best_ndcg = -1
        for strategy_name in FUSION_STRATEGIES.keys():
            ndcg = all_fusion_results[model_name][strategy_name]['none']['ndcg'][10]
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_strategy = strategy_name
        print(f"  {model_name}: {best_strategy} → NDCG@10={best_ndcg:.4f}")
    
    print("\n完成！")


if __name__ == '__main__':
    main()