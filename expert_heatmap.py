#!/usr/bin/env python3
"""
expert_heatmap.py - 从 checkpoint 提取每任务×每专家的 gate 权重，生成专家激活热力图
不需要重新训练，只需加载 checkpoint + 前向推理

用法: python3 expert_heatmap.py [--max-batches 100]
输出: results/analysis/expert_heatmap_*.png
"""
import sys, os, json, re
import numpy as np
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    from torch import nn
except ImportError:
    print("ERROR: torch required")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("ERROR: matplotlib required")
    sys.exit(1)

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = '/root/mtl4rec'
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'analysis')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints', 'KuaiRand1K')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
         'long_view', 'is_hate', 'is_profile_enter']
TASK_NAMES = ['Click', 'Like', 'Follow', 'Comment', 'Forward',
              'LongView', 'Hate', 'Profile']

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'fuxictr'))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def collect_gate_hooks(model):
    """注册 forward hook 收集所有 gate 层输出"""
    gate_data = defaultdict(list)
    hooks = []

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 2:
                gate_data[layer_name].append(output.detach().cpu().numpy())
        return hook_fn

    for name, module in model.named_modules():
        # CGC gates (PLE)
        if re.search(r'cgc_layers\.\d+\.gate\.\d+$', name):
            if 'gate_net' not in name and 'gate_activation' not in name:
                hook = module.register_forward_hook(make_hook(f"cgc_{name}"))
                hooks.append(hook)
        # MMoE gates
        if re.search(r'mmoe_layer\.gate\.\d+$', name):
            hook = module.register_forward_hook(make_hook(f"mmoe_{name}"))
            hooks.append(hook)

    return gate_data, hooks


def build_per_task_gate(gate_data, model_type, num_tasks, num_total_experts,
                        num_shared=None, num_specific=None):
    """
    将收集的 gate logits 组织为 per-task 的 gate 权重矩阵
    PLE: gate[0..7] = task gates, gate[8] = shared gate
    MMoE: gate[0..7] = task gates
    返回: (num_tasks, num_total_experts) 的平均 gate 权重
    """
    if model_type == 'PLE':
        # CGC: gate[0..7] are task-specific, each gate outputs (num_shared + num_specific) logits
        task_gates = []
        for t in range(num_tasks):
            key = f"cgc_cgc_layers.0.gate.{t}"
            if key in gate_data:
                logits = np.concatenate(gate_data[key], axis=0)
                probs = softmax(logits, axis=1)
                task_gates.append(probs.mean(axis=0))
            else:
                task_gates.append(np.zeros(num_total_experts))
        return np.array(task_gates)

    elif model_type == 'MMoE':
        task_gates = []
        for t in range(num_tasks):
            key = f"mmoe_mmoe_layer.gate.{t}"
            if key in gate_data:
                logits = np.concatenate(gate_data[key], axis=0)
                probs = softmax(logits, axis=1)
                task_gates.append(probs.mean(axis=0))
            else:
                task_gates.append(np.zeros(num_total_experts))
        return np.array(task_gates)

    return None


def analyze_model(model_name, config_expid, checkpoint_path, max_batches=100):
    """加载 checkpoint，提取 per-task gate 权重"""
    print(f"\n  [Analyzing] {model_name}...")

    from fuxictr.utils import load_config
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.dataloaders import H5DataLoader
    import models as model_zoo

    params = load_config(CONFIG_DIR, config_expid)
    params['gpu'] = -1

    for key in ['train_data', 'valid_data', 'test_data']:
        if key in params and isinstance(params[key], str) and params[key].endswith('.csv'):
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

    # Determine model type
    if 'PLE' in model_name or 'PLE' in config_expid:
        model_type = 'PLE'
    elif 'MMoE' in model_name or 'MMoE' in config_expid:
        model_type = 'MMoE'
    else:
        model_type = 'SharedBottom'

    # Run inference
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

    print(f"    Captured {len(gate_data)} gate layers, {batch_count} batches")

    # Build per-task gate matrix
    num_shared = params.get('num_shared_experts', 4)
    num_specific = params.get('num_specific_experts', 2)
    num_total = num_shared + num_specific
    num_tasks = len(TASKS)

    gate_matrix = build_per_task_gate(
        gate_data, model_type, num_tasks, num_total,
        num_shared=num_shared, num_specific=num_specific
    )

    return {
        'model_name': model_name,
        'model_type': model_type,
        'gate_matrix': gate_matrix,
        'num_shared': num_shared,
        'num_specific': num_specific,
        'num_total': num_total
    }


def plot_single_heatmap(gate_matrix, model_name, model_type, num_shared, num_specific, output_path):
    """为单个模型画专家激活热力图"""
    num_tasks = gate_matrix.shape[0]
    num_experts = gate_matrix.shape[1]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Custom colormap: white for uniform, red for concentrated
    cmap = LinearSegmentedColormap.from_list('gate_cmap',
        ['#F7F7F7', '#FEE090', '#E08214', '#B2182B'], N=256)

    im = ax.imshow(gate_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=gate_matrix.max() * 1.1)

    # Annotate each cell
    for i in range(num_tasks):
        for j in range(num_experts):
            val = gate_matrix[i, j]
            text_color = 'white' if val > gate_matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, color=text_color, fontweight='bold')

    # Expert labels
    if model_type == 'PLE':
        expert_labels = [f'Shared{i+1}' for i in range(num_shared)] + \
                        [f'Specific{i+1}' for i in range(num_specific)]
    else:
        expert_labels = [f'Expert{i+1}' for i in range(num_experts)]

    ax.set_xticks(range(num_experts))
    ax.set_xticklabels(expert_labels, fontsize=10)
    ax.set_yticks(range(num_tasks))
    ax.set_yticklabels(TASK_NAMES, fontsize=11)

    # Highlight worst task
    worst_task_idx = int(np.argmin(gate_matrix.max(axis=1)))
    for j in range(num_experts):
        ax.add_patch(plt.Rectangle((j - 0.5, worst_task_idx - 0.5), 1, 1,
                                    fill=False, edgecolor='#00FF00', linewidth=2.5, linestyle='--'))

    ax.set_title(f'{model_name} - Expert Gate Weights\n(Green dashed = most marginalized task)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Experts', fontsize=12)
    ax.set_ylabel('Tasks', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Gate Weight', fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_path}")


def plot_comparison_heatmap(all_results, output_path):
    """对比热力图：每个模型一行，显示各任务对专家的平均利用"""
    if not all_results:
        return

    # Filter to models with gate data
    models_with_gates = [(k, v) for k, v in sorted(all_results.items())
                         if v['model_type'] in ('PLE', 'MMoE') and v['gate_matrix'] is not None]

    if not models_with_gates:
        return

    fig, axes = plt.subplots(len(models_with_gates), 1, figsize=(16, 3.5 * len(models_with_gates)))
    if len(models_with_gates) == 1:
        axes = [axes]

    for ax, (model_name, info) in zip(axes, models_with_gates):
        gate_matrix = info['gate_matrix']
        num_shared = info['num_shared']
        num_specific = info['num_specific']
        model_type = info['model_type']

        cmap = LinearSegmentedColormap.from_list('gate_cmap',
            ['#F7F7F7', '#FEE090', '#E08214', '#B2182B'], N=256)

        im = ax.imshow(gate_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=0.35)

        for i in range(gate_matrix.shape[0]):
            for j in range(gate_matrix.shape[1]):
                val = gate_matrix[i, j]
                text_color = 'white' if val > 0.22 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, color=text_color, fontweight='bold')

        if model_type == 'PLE':
            expert_labels = [f'Shared{i+1}' for i in range(num_shared)] + \
                            [f'Spec{i+1}' for i in range(num_specific)]
        else:
            expert_labels = [f'Expert{i+1}' for i in range(gate_matrix.shape[1])]

        ax.set_xticks(range(gate_matrix.shape[1]))
        ax.set_xticklabels(expert_labels, fontsize=9)
        ax.set_yticks(range(len(TASK_NAMES)))
        ax.set_yticklabels(TASK_NAMES, fontsize=10)
        ax.set_title(f'{model_name} ({model_type})', fontsize=12, fontweight='bold')

    plt.suptitle('Expert Activation Heatmap Comparison\n(Each row = one model, columns = experts, rows within = tasks)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_path}")


def plot_expert_utilization_summary(all_results, output_path):
    """汇总：每个模型的专家利用率 + 最边缘化任务"""
    models_with_gates = [(k, v) for k, v in sorted(all_results.items())
                         if v['model_type'] in ('PLE', 'MMoE') and v['gate_matrix'] is not None]

    if not models_with_gates:
        return

    fig, axes = plt.subplots(1, len(models_with_gates), figsize=(4 * len(models_with_gates), 6))
    if len(models_with_gates) == 1:
        axes = [axes]

    for ax, (model_name, info) in zip(axes, models_with_gates):
        gate_matrix = info['gate_matrix']
        num_tasks = gate_matrix.shape[0]

        # Per-task: max gate weight vs average gate weight
        max_gate = gate_matrix.max(axis=1)
        avg_gate = gate_matrix.mean(axis=1)
        # Entropy per task
        eps = 1e-8
        clamped = np.clip(gate_matrix, eps, 1.0)
        entropy = -np.sum(clamped * np.log(clamped), axis=1)
        max_entropy = np.log(gate_matrix.shape[1])
        norm_entropy = entropy / max_entropy

        x = np.arange(num_tasks)
        width = 0.25

        bars1 = ax.bar(x - width, max_gate, width, label='Max Gate', color='#E74C3C', edgecolor='white')
        bars2 = ax.bar(x, avg_gate, width, label='Avg Gate', color='#3498DB', edgecolor='white')
        bars3 = ax.bar(x + width, norm_entropy, width, label='Norm Entropy', color='#2ECC71', edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(TASK_NAMES, rotation=30, ha='right', fontsize=9)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gate Weight / Entropy', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Task Expert Utilization Summary\n(Max Gate = dominant expert, Entropy = diversity)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-batches', type=int, default=100)
    parser.add_argument('--models', nargs='+', default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Models to analyze: the progression story
    # SharedBottom < MMoE < MMoE_v2 < PLE < PLE_v13 < PLE_v14 < PLE_v15
    if args.models:
        models_to_analyze = []
        for m in args.models:
            models_to_analyze.append((m, f"{m}_KuaiRand1K", f"{m}_KuaiRand1K.model"))
    else:
        models_to_analyze = [
            # Baselines
            ('SharedBottom', 'SharedBottom_KuaiRand1K', 'SharedBottom_KuaiRand1K.model'),
            ('MMoE_base', 'MMoE_KuaiRand1K', 'MMoE_KuaiRand1K.model'),
            ('MMoE_v2', 'MMoE_v2_KuaiRand1K', 'MMoE_v2_KuaiRand1K.model'),
            # PLE progression
            ('PLE_v5', 'PLE_KuaiRand1K', 'PLE_KuaiRand1K.model'),
            ('PLE_v13', 'PLE_v13_KuaiRand1K', 'PLE_v13_KuaiRand1K.model'),
            ('PLE_v14', 'PLE_v14_KuaiRand1K', 'PLE_v14_KuaiRand1K.model'),
            ('PLE_v15', 'PLE_v15_KuaiRand1K', 'PLE_v15_KuaiRand1K.model'),
        ]

    print("=" * 70)
    print("  Expert Activation Heatmap Analysis")
    print("=" * 70)

    all_results = {}

    for model_name, expid, ckpt_file in models_to_analyze:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {model_name}: {ckpt_file} not found")
            continue

        try:
            result = analyze_model(model_name, expid, ckpt_path, args.max_batches)
            all_results[model_name] = result

            # Individual heatmap per model
            if result['gate_matrix'] is not None:
                plot_single_heatmap(
                    result['gate_matrix'], model_name, result['model_type'],
                    result['num_shared'], result['num_specific'],
                    os.path.join(OUTPUT_DIR, f'expert_heatmap_{model_name}.png')
                )
        except Exception as e:
            print(f"  ERROR {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Comparison visualizations
    plot_comparison_heatmap(all_results, os.path.join(OUTPUT_DIR, 'expert_heatmap_comparison.png'))
    plot_expert_utilization_summary(all_results, os.path.join(OUTPUT_DIR, 'expert_utilization_summary.png'))

    # Save numerical data
    numeric_data = {}
    for model_name, info in all_results.items():
        if info['gate_matrix'] is not None:
            numeric_data[model_name] = {
                'model_type': info['model_type'],
                'gate_matrix': info['gate_matrix'].tolist(),
                'task_names': TASK_NAMES,
                'num_shared': info['num_shared'],
                'num_specific': info['num_specific']
            }

    with open(os.path.join(OUTPUT_DIR, 'expert_gate_data.json'), 'w') as f:
        json.dump(numeric_data, f, indent=2, ensure_ascii=False)
    print(f"\n  [SAVED] expert_gate_data.json")

    print(f"\n{'='*70}")
    print(f"  Done! {len(all_results)} models analyzed.")
    print(f"  Files in {OUTPUT_DIR}:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if 'expert' in f:
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"    {f} ({size_kb:.1f} KB)")
    print("=" * 70)


if __name__ == '__main__':
    main()