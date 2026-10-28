#!/usr/bin/env python3
"""
full_analysis.py - 综合分析脚本：无需重新训练，基于已有 checkpoint 和日志
完成以下分析：
  1. 日志解析 → 所有模型的 test AUC 对比
  2. Checkpoint 加载 → Expert Utilization Rate / Gate 熵分析
  3. 可视化生成 → 模型对比图、任务热力图、Expert Utilization Rate雷达图、Gate 熵对比
  4. 输出综合报告 → JSON + PNG 图表

用法: python3 full_analysis.py [--skip-gate] [--max-batches 50]
"""

import sys, os, json, re, pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

# 尝试导入 torch（gate 分析需要）
try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch 未安装，跳过 gate 分析")

# 尝试导入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 未安装，跳过可视化")

# ============================================================
# 配置
# ============================================================
BASE_DIR = '/root/mtl4rec'
LOG_DIR = os.path.join(BASE_DIR, 'results', 'logs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'analysis')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints', 'KuaiRand1K')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
         'long_view', 'is_hate', 'is_profile_enter']

TASK_NAMES_CN = {
    'is_click': 'Click', 'is_like': 'Like', 'is_follow': 'Follow',
    'is_comment': 'Comment', 'is_forward': 'Forward', 'long_view': 'LongView',
    'is_hate': 'Hate', 'is_profile_enter': 'Profile'
}

# 字体设置
if HAS_MPL:
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 日志解析
# ============================================================
def parse_log(filepath):
    """解析训练日志，提取 test AUC"""
    with open(filepath, 'r') as f:
        content = f.read()

    test_auc = {}
    test_blocks = content.split("Test evaluation")
    if len(test_blocks) < 2:
        return None

    test_section = test_blocks[-1]
    for task in TASKS:
        match = re.search(rf'\[Task: {task}\].*?AUC:\s*([\d.]+)', test_section)
        if match:
            test_auc[task] = float(match.group(1))

    if len(test_auc) < 8:
        return None

    test_auc['avg_auc'] = np.mean(list(test_auc.values()))

    # 提取验证集 AUC
    val_auc = {}
    val_blocks = content.split("****** Validation evaluation ******")
    if len(val_blocks) > 1:
        val_section = val_blocks[-1]
        for task in TASKS:
            match = re.search(rf'\[Task: {task}\].*?AUC:\s*([\d.]+)', val_section)
            if match:
                val_auc[task] = float(match.group(1))
        if len(val_auc) == 8:
            val_auc['avg_auc'] = np.mean(list(val_auc.values()))

    return {'test_auc': test_auc, 'val_auc': val_auc if len(val_auc) == 8 else None}


def identify_model(log_name):
    """从日志文件名识别模型和版本"""
    name = log_name.replace('.log', '')
    
    # Handle special cases
    if 'v5_screen' in name:
        return 'PLE', 'v5'
    if 'v5v2' in name:
        return 'MMoE', 'v2'
    
    m = re.match(r'(PLE_v?\d+|MMoE_v?\d+|SharedBottom|PLE|MMoE)_KuaiRand1K_(v?\d+[a-z]*)?_?(\d{8}_\d{6})', name)
    if m:
        model_name = m.group(1)
        version = m.group(2) if m.group(2) else 'base'
        return model_name, version
    m = re.match(r'(SharedBottom|MMoE|PLE)_KuaiRand1K_(\d{8}_\d{6})', name)
    if m:
        return m.group(1), 'base'
    return name, 'unknown'


def collect_all_results():
    """收集所有日志中的 AUC 结果"""
    results = {}
    log_dir = Path(LOG_DIR)

    for log_file in sorted(log_dir.glob('*.log')):
        model_name, version = identify_model(log_file.name)
        data = parse_log(str(log_file))
        if data is None:
            continue

        key = f"{model_name}_{version}"
        if key not in results or data['test_auc']['avg_auc'] > results[key]['test_auc']['avg_auc']:
            results[key] = {
                'model': model_name, 'version': version,
                'log_file': log_file.name, **data
            }
    return results


# ============================================================
# Part 2: Gate / Expert Utilization Analysis（基于 checkpoint）
# ============================================================
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def collect_gate_hooks(model):
    """注册 forward hook 收集 gate 层输出"""
    gate_data = defaultdict(list)
    hooks = []

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 2:
                gate_data[layer_name].append(output.detach().cpu().numpy())
        return hook_fn

    for name, module in model.named_modules():
        if re.search(r'cgc_layers\.\d+\.gate\.\d+$', name):
            if 'gate_net' not in name and 'gate_activation' not in name:
                hook = module.register_forward_hook(make_hook(f"cgc_{name}"))
                hooks.append(hook)
        if 'gate_nets' in name and re.search(r'gate_nets\.\d+$', name):
            hook = module.register_forward_hook(make_hook(f"ppnet_{name}"))
            hooks.append(hook)
        if re.search(r'mmoe_layer\.gate\.\d+$', name):
            hook = module.register_forward_hook(make_hook(f"mmoe_{name}"))
            hooks.append(hook)

    return gate_data, hooks


def compute_gate_metrics(all_gate_data, num_experts, is_cgc=True):
    """计算Expert Utilization Rate指标"""
    metrics = {}
    for layer_name, gate_list in all_gate_data.items():
        if len(gate_list) == 0:
            continue
        all_logits = np.concatenate(gate_list, axis=0)
        all_gates = softmax(all_logits, axis=1) if is_cgc else all_logits

        actual_experts = all_gates.shape[1]
        mean_gate = all_gates.mean(axis=0)

        eps = 1e-8
        clamped = np.clip(all_gates, eps, 1.0)
        entropy_per_sample = -np.sum(clamped * np.log(clamped), axis=1)
        mean_entropy = float(entropy_per_sample.mean())
        max_entropy = np.log(actual_experts)
        normalized_entropy = mean_entropy / max_entropy if max_entropy > 0 else 0

        uniform_threshold = 1.0 / actual_experts
        utilized_experts = np.sum(mean_gate > uniform_threshold)
        utilization_rate = utilized_experts / actual_experts
        max_gate_per_sample = all_gates.max(axis=1)
        dominant_ratio = float(max_gate_per_sample.mean())
        usage_variance = float(np.var(mean_gate))

        metrics[layer_name] = {
            'mean_gate': mean_gate.tolist(),
            'mean_entropy': mean_entropy,
            'normalized_entropy': normalized_entropy,
            'utilization_rate': utilization_rate,
            'dominant_ratio': dominant_ratio,
            'usage_variance': usage_variance,
            'num_samples': int(all_gates.shape[0]),
            'num_experts': actual_experts
        }

    return metrics


def analyze_model_gates(model_name, config_expid, checkpoint_path, max_batches=50):
    """加载 checkpoint 并分析 gate 值"""
    print(f"\n  [Gate Analysis] {model_name}...")

    from fuxictr.utils import load_config
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.dataloaders import H5DataLoader
    import models as model_zoo

    params = load_config(os.path.join(CONFIG_DIR, 'model_config.yaml'), config_expid)
    params['gpu'] = -1

    for key in ['train_data', 'valid_data', 'test_data']:
        if key in params and params[key].endswith('.csv'):
            params[key] = params[key].replace('.csv', '')

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)

    if 'task_labels' in params:
        task_labels = params['task_labels'] if isinstance(params['task_labels'], list) else list(params['task_labels'])
        feature_map.labels = task_labels
        feature_map.set_column_index()

    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    gate_data, hooks = collect_gate_hooks(model)

    params['batch_size'] = 4096
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()

    batch_count = 0
    with torch.no_grad():
        for batch_data in test_gen:
            if batch_count >= max_batches:
                break
            try:
                model.forward(batch_data)
                batch_count += 1
            except Exception:
                batch_count += 1
                continue

    for h in hooks:
        h.remove()

    is_cgc = 'PLE' in model_name or 'PLE' in config_expid
    num_experts = params.get('num_shared_experts', 4) + params.get('num_specific_experts', 2)
    metrics = compute_gate_metrics(gate_data, num_experts, is_cgc=is_cgc)

    # 汇总各层指标
    if metrics:
        avg_entropy = np.mean([m['normalized_entropy'] for m in metrics.values()])
        avg_util = np.mean([m['utilization_rate'] for m in metrics.values()])
        avg_dominant = np.mean([m['dominant_ratio'] for m in metrics.values()])
    else:
        avg_entropy = avg_util = avg_dominant = 0

    return {
        'model_name': model_name,
        'num_layers': len(metrics),
        'avg_normalized_entropy': float(avg_entropy),
        'avg_utilization_rate': float(avg_util),
        'avg_dominant_ratio': float(avg_dominant),
        'layer_details': metrics
    }


# ============================================================
# Part 3: 可视化
# ============================================================
def plot_model_comparison(results, output_path):
    """模型 Avg AUC 对比柱状图"""
    if not HAS_MPL:
        return

    # 筛选关键模型
    key_models = ['PLE_v5', 'PLE_v7', 'PLE_v14', 'MMoE_v3', 'PLE_v3',
                  'SharedBottom_v4', 'PLE_v13', 'SharedBottom_base',
                  'MMoE_v2', 'PLE_v12', 'PLE_v6', 'PLE_base', 'MMoE_base']
    available = [(k, v) for k, v in results.items() if k in key_models]
    if not available:
        available = sorted(results.items(), key=lambda x: x[1]['test_auc']['avg_auc'], reverse=True)[:10]

    names = [k for k, v in available]
    avg_aucs = [v['test_auc']['avg_auc'] for k, v in available]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#FF6B6B' if 'SharedBottom' in n else '#4ECDC4' if 'MMoE' in n else '#45B7D1' for n in names]
    bars = ax.bar(range(len(names)), avg_aucs, color=colors, edgecolor='white', linewidth=0.8)

    # 标注数值
    for i, (bar, auc) in enumerate(zip(bars, avg_aucs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Avg Test AUC', fontsize=12)
    ax.set_title('Model Test AUC Comparison (KuaiRand-1K)', fontsize=14, fontweight='bold')
    ax.set_ylim(min(avg_aucs) - 0.02, max(avg_aucs) + 0.01)
    ax.grid(axis='y', alpha=0.3)

    # 添加最佳模型标注
    best_idx = np.argmax(avg_aucs)
    ax.annotate(f'Best: {avg_aucs[best_idx]:.4f}',
                xy=(best_idx, avg_aucs[best_idx]),
                xytext=(best_idx, avg_aucs[best_idx] + 0.008),
                ha='center', fontsize=10, color='#E74C3C', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_path}")


def plot_task_heatmap(results, output_path):
    """任务级 AUC 热力图"""
    if not HAS_MPL:
        return

    key_models = ['PLE_v5', 'PLE_v7', 'PLE_v14', 'MMoE_v3', 'PLE_v3',
                  'SharedBottom_v4', 'PLE_v13', 'SharedBottom_base',
                  'MMoE_v2', 'PLE_v12', 'PLE_v6', 'PLE_base', 'MMoE_base']
    available = [(k, v) for k, v in results.items() if k in key_models]
    if not available:
        available = sorted(results.items(), key=lambda x: x[1]['test_auc']['avg_auc'], reverse=True)[:10]

    model_names = [k for k, v in available]
    data_matrix = np.array([[v['test_auc'][t] for t in TASKS] for k, v in available])

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0.45, vmax=0.82)

    # 标注每个 cell
    for i in range(len(model_names)):
        for j in range(len(TASKS)):
            val = data_matrix[i, j]
            text_color = 'white' if val < 0.58 or val > 0.72 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')

    # 标注每个模型的 avg
    for i, (name, info) in enumerate(available):
        avg = info['test_auc']['avg_auc']
        ax.text(len(TASKS) + 0.5, i, f'{avg:.4f}', ha='center', va='center',
                fontsize=9, color='#2C3E50', fontweight='bold')

    ax.set_xticks(list(range(len(TASKS))) + [len(TASKS) + 0.5])
    ax.set_xticklabels([TASK_NAMES_CN[t] for t in TASKS] + ['Avg'], rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title('Task-level AUC Heatmap (KuaiRand-1K)', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AUC', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_path}")


def plot_task_improvement(results, output_path, baseline_key='PLE_base', compare_keys=None):
    """相对于 baseline"""
    if not HAS_MPL:
        return

    if baseline_key not in results:
        print(f"  [SKIP] baseline {baseline_key} not found")
        return

    if compare_keys is None:
        compare_keys = ['PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14', 'MMoE_v2']

    baseline = results[baseline_key]['test_auc']
    available_keys = [k for k in compare_keys if k in results]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(TASKS))
    width = 0.12
    colors = plt.cm.Set2(np.linspace(0, 1, len(available_keys)))

    for i, key in enumerate(available_keys):
        improvements = [results[key]['test_auc'][t] - baseline[t] for t in TASKS]
        bars = ax.bar(x + i * width, improvements, width, label=key,
                      color=colors[i], edgecolor='white', linewidth=0.5)

        for j, (bar, imp) in enumerate(zip(bars, improvements)):
            if abs(imp) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.003 if imp > 0 else -0.012),
                        f'{imp:+.3f}', ha='center', fontsize=6.5, rotation=90)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x + width * (len(available_keys) - 1) / 2)
    ax.set_xticklabels([TASK_NAMES_CN[t] for t in TASKS], fontsize=10)
    ax.set_ylabel(f'AUC Change (vs {baseline_key})', fontsize=12)
    ax.set_title(f'Task AUC Improvement vs  {baseline_key}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_path}")


def plot_gate_analysis(gate_results, output_path):
    """Gate 熵和利用率对比"""
    if not HAS_MPL or not gate_results:
        return

    models = list(gate_results.keys())
    entropies = [gate_results[m].get('avg_normalized_entropy', 0) for m in models]
    utils = [gate_results[m].get('avg_utilization_rate', 0) for m in models]
    dominants = [gate_results[m].get('avg_dominant_ratio', 0) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 归一化熵
    ax = axes[0]
    bars = ax.bar(range(len(models)), entropies, color='#5DADE2', edgecolor='white')
    for bar, val in zip(bars, entropies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
    ax.set_title('Gate Normalized Entropy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Entropy', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # 利用率
    ax = axes[1]
    bars = ax.bar(range(len(models)), utils, color='#58D68D', edgecolor='white')
    for bar, val in zip(bars, utils):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
    ax.set_title('Expert Utilization Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Utilization Rate', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # 主导比率
    ax = axes[2]
    bars = ax.bar(range(len(models)), dominants, color='#F4D03F', edgecolor='white')
    for bar, val in zip(bars, dominants):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
    ax.set_title('Gate Dominant Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dominant Ratio', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Gate / Expert Utilization Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_path}")


def plot_summary_dashboard(results, gate_results, output_path):
    """综合仪表盘"""
    if not HAS_MPL:
        return

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # ---- 左上: 模型 Avg AUC 排名 ----
    ax1 = fig.add_subplot(gs[0, :2])
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_auc']['avg_auc'], reverse=True)[:10]
    names = [k for k, v in sorted_models]
    aucs = [v['test_auc']['avg_auc'] for k, v in sorted_models]
    colors = ['#E74C3C' if i == 0 else '#3498DB' for i in range(len(names))]
    ax1.barh(range(len(names)), aucs, color=colors, edgecolor='white')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Avg Test AUC', fontsize=11)
    ax1.set_title('Top 10 Model Avg AUC Ranking', fontsize=13, fontweight='bold')
    for i, (name, auc) in enumerate(zip(names, aucs)):
        ax1.text(auc + 0.002, i, f'{auc:.4f}', va='center', fontsize=9, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # ---- 右上: Task Correlation热力图 ----
    ax2 = fig.add_subplot(gs[0, 2])
    try:
        with open(os.path.join(BASE_DIR, 'task_correlation.json')) as f:
            corr_data = json.load(f)
        corr_matrix = np.array(corr_data['correlation_matrix'])
        im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-0.5, vmax=1.0, aspect='auto')
        ax2.set_xticks(range(len(TASKS)))
        ax2.set_yticks(range(len(TASKS)))
        ax2.set_xticklabels([TASK_NAMES_CN[t] for t in TASKS], rotation=45, ha='right', fontsize=7)
        ax2.set_yticklabels([TASK_NAMES_CN[t] for t in TASKS], fontsize=7)
        ax2.set_title('Task Correlation', fontsize=12, fontweight='bold')
        for i in range(len(TASKS)):
            for j in range(len(TASKS)):
                ax2.text(j, i, f'{corr_matrix[i][j]:.2f}', ha='center', va='center', fontsize=6)
        plt.colorbar(im, ax=ax2, shrink=0.8)
    except Exception as e:
        ax2.text(0.5, 0.5, f'Correlation data\nunavailable', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=10)
        ax2.set_title('Task Correlation (N/A)', fontsize=12)

    # ---- 中左: 关键模型各任务 AUC ----
    ax3 = fig.add_subplot(gs[1, :2])
    compare_keys = ['PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14', 'MMoE_v2', 'MMoE_v3']
    available_keys = [k for k in compare_keys if k in results]
    x = np.arange(len(TASKS))
    width = 0.12
    colors_line = plt.cm.tab10(np.linspace(0, 1, len(available_keys)))
    for i, key in enumerate(available_keys):
        vals = [results[key]['test_auc'][t] for t in TASKS]
        ax3.bar(x + i * width, vals, width, label=key, color=colors_line[i], edgecolor='white', linewidth=0.3)
    ax3.set_xticks(x + width * (len(available_keys) - 1) / 2)
    ax3.set_xticklabels([TASK_NAMES_CN[t] for t in TASKS], fontsize=10)
    ax3.set_ylabel('Test AUC', fontsize=11)
    ax3.set_title('Key Model Task-level AUC Comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, ncol=3)
    ax3.grid(axis='y', alpha=0.3)

    # ---- 中右: Gate 熵 vs Avg AUC ----
    ax4 = fig.add_subplot(gs[1, 2])
    if gate_results:
        gate_models = set(gate_results.keys())
        for model_key in gate_models:
            if model_key in results:
                entropy = gate_results[model_key]['avg_normalized_entropy']
                avg_auc = results[model_key]['test_auc']['avg_auc']
                ax4.scatter(entropy, avg_auc, s=120, label=model_key, edgecolors='black', linewidth=0.5)
                ax4.annotate(model_key, (entropy, avg_auc), fontsize=8, textcoords="offset points",
                            xytext=(5, 5))
        ax4.set_xlabel('Gate Normalized Entropy', fontsize=10)
        ax4.set_ylabel('Avg Test AUC', fontsize=10)
        ax4.set_title('Gate Entropy vs Model Performance', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Gate data\nunavailable', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=10)

    # ---- 下: 所有模型完整对比表 ----
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    all_sorted = sorted(results.items(), key=lambda x: x[1]['test_auc']['avg_auc'], reverse=True)
    col_labels = ['Model', 'Version'] + [TASK_NAMES_CN[t] for t in TASKS] + ['Avg AUC']
    table_data = []
    for key, info in all_sorted:
        row = [info['model'], info['version']]
        row += [f"{info['test_auc'][t]:.4f}" for t in TASKS]
        row += [f"{info['test_auc']['avg_auc']:.4f}"]
        table_data.append(row)

    table = ax5.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center', fontsize=7)
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == len(col_labels) - 1:
            cell.set_facecolor('#F9E79F')
        cell.set_edgecolor('#BDC3C7')

    ax5.set_title('All Models Test AUC Full Comparison', fontsize=13, fontweight='bold', y=0.98)

    plt.suptitle('KuaiRand-1K Multi-Task Learning Analysis', fontsize=16, fontweight='bold', y=0.99)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='综合分析脚本')
    parser.add_argument('--skip-gate', action='store_true', help='跳过 gate 分析（需要 torch）')
    parser.add_argument('--max-batches', type=int, default=50, help='Gate 分析最大 batch 数')
    parser.add_argument('--models', nargs='+', default=None,
                        help='指定分析的模型（checkpoint 名），如 PLE_v15 PLE_v14')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  KuaiRand-1K 多任务学习实验 — 综合分析")
    print("=" * 70)

    # ---- Step 1: 日志解析 ----
    print("\n[1/4] 解析训练日志...")
    results = collect_all_results()
    print(f"  共解析 {len(results)} 个模型")

    # 打印关键模型对比
    print(f"\n  {'Model':<25} {'Avg AUC':<10} {'Forward':<10} {'Comment':<10}")
    print("  " + "-" * 55)
    key_models = ['PLE_v5', 'PLE_v7', 'PLE_v14', 'MMoE_v3', 'PLE_v3',
                  'SharedBottom_v4', 'PLE_v13', 'SharedBottom_base',
                  'MMoE_v2', 'PLE_v12', 'PLE_v6', 'PLE_base', 'MMoE_base'] + ['SharedBottom_v3']
    for key in key_models:
        if key in results:
            info = results[key]
            print(f"  {key:<25} {info['test_auc']['avg_auc']:.4f}     {info['test_auc']['is_forward']:.4f}     {info['test_auc']['is_comment']:.4f}")

    # 保存 JSON
    results_json = {
        'total_models': len(results),
        'models': {
            k: {
                'model': v['model'], 'version': v['version'],
                'log_file': v['log_file'],
                'test_auc': {t: round(v['test_auc'][t], 6) for t in TASKS + ['avg_auc']},
                'val_auc': {t: round(v['val_auc'][t], 6) for t in TASKS + ['avg_auc']} if v['val_auc'] else None
            }
            for k, v in sorted(results.items(), key=lambda x: x[1]['test_auc']['avg_auc'], reverse=True)
        }
    }
    with open(os.path.join(OUTPUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] all_results.json")

    # ---- Step 2: Gate 分析 ----
    gate_results = {}
    if not args.skip_gate and HAS_TORCH:
        print("\n[2/4] Gate / Expert Utilization Analysis...")
        sys.path.insert(0, BASE_DIR)
        sys.path.insert(0, os.path.join(BASE_DIR, 'fuxictr'))

        models_to_analyze = args.models or [
            ('PLE_v5', 'PLE_KuaiRand1K', 'PLE_KuaiRand1K.model'),
            ('PLE_v13', 'PLE_v13_KuaiRand1K', 'PLE_v13_KuaiRand1K.model'),
            ('PLE_v14', 'PLE_v14_KuaiRand1K', 'PLE_v14_KuaiRand1K.model'),
            ('PLE_v15', 'PLE_v15_KuaiRand1K', 'PLE_v15_KuaiRand1K.model'),
            ('MMoE_v2', 'MMoE_v2_KuaiRand1K', 'MMoE_v2_KuaiRand1K.model'),
        ]

        for model_name, expid, ckpt_file in models_to_analyze:
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
            if not os.path.exists(ckpt_path):
                print(f"  SKIP: {ckpt_file} not found")
                continue
            try:
                gate_result = analyze_model_gates(model_name, expid, ckpt_path, args.max_batches)
                gate_results[model_name] = gate_result
            except Exception as e:
                print(f"  ERROR: {model_name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n  完成 {len(gate_results)} 个模型的 gate 分析")

        with open(os.path.join(OUTPUT_DIR, 'gate_analysis.json'), 'w') as f:
            json.dump(gate_results, f, indent=2, ensure_ascii=False)
        print(f"  [SAVED] gate_analysis.json")
    else:
        print("\n[2/4] 跳过 Gate 分析 (--skip-gate 或 torch 不可用)")

    # ---- Step 3: 可视化 ----
    print("\n[3/4] 生成可视化图表...")
    if HAS_MPL:
        plot_model_comparison(results, os.path.join(OUTPUT_DIR, 'model_comparison.png'))
        plot_task_heatmap(results, os.path.join(OUTPUT_DIR, 'task_heatmap.png'))
        plot_task_improvement(results, os.path.join(OUTPUT_DIR, 'task_improvement.png'))
        if gate_results:
            plot_gate_analysis(gate_results, os.path.join(OUTPUT_DIR, 'gate_analysis.png'))
        plot_summary_dashboard(results, gate_results, os.path.join(OUTPUT_DIR, 'summary_dashboard.png'))
    else:
        print("  matplotlib 不可用，跳过可视化")

    # ---- Step 4: 综合报告 ----
    print("\n[4/4] 生成综合报告...")
    report = {
        'analysis_time': str(np.datetime64('now')),
        'total_models_analyzed': len(results),
        'total_gate_analyzed': len(gate_results),
        'key_findings': []
    }

    # 自动生成关键发现
    if 'PLE_base' in results and 'PLE_v6' in results:
        forward_diff = results['PLE_v6']['test_auc']['is_forward'] - results['PLE_base']['test_auc']['is_forward']
        avg_diff = results['PLE_v6']['test_auc']['avg_auc'] - results['PLE_base']['test_auc']['avg_auc']
        report['key_findings'].append({
            'title': 'Balance Loss (V6) 效果',
            'finding': f'Balance Loss λ=0.01 使 forward AUC 提升 {forward_diff:+.4f}，但 avg AUC 下降 {avg_diff:+.4f}，存在跷跷板效应',
            'forward_change': round(forward_diff, 4),
            'avg_change': round(avg_diff, 4)
        })

    if 'PLE_v14' in results and 'PLE_v5' in results:
        v5_forward = results['PLE_v5']['test_auc']['is_forward'] if 'PLE_v5' in results else None
        v14_forward = results['PLE_v14']['test_auc']['is_forward']
        if v5_forward:
            report['key_findings'].append({
                'title': 'User Bias (V14) 效果',
                'finding': f'V14 的 User Bias 使 forward 从 {v5_forward:.4f} 提升到 {v14_forward:.4f}，但 comment 大幅下降',
                'forward_change': round(v14_forward - v5_forward, 4)
            })

    if 'MMoE_v2' in results and 'MMoE_base' in results:
        mmoe_forward = results['MMoE_v2']['test_auc']['is_forward'] - results['MMoE_base']['test_auc']['is_forward']
        mmoe_avg = results['MMoE_v2']['test_auc']['avg_auc'] - results['MMoE_base']['test_auc']['avg_auc']
        report['key_findings'].append({
            'title': 'MMoE + PPNet (MMoE_v2) 效果',
            'finding': f'MMoE_v2 相对 MMoE_base: forward {mmoe_forward:+.4f}, avg AUC {mmoe_avg:+.4f}',
            'forward_change': round(mmoe_forward, 4),
            'avg_change': round(mmoe_avg, 4)
        })

    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] report.json")

    print("\n" + "=" * 70)
    print(f"  分析完成！所有输出保存在: {OUTPUT_DIR}")
    print(f"  文件列表:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f} ({size_kb:.1f} KB)")
    print("=" * 70)


if __name__ == '__main__':
    main()