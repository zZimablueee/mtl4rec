
#!/usr/bin/env python3
"""
analyze_gates.py - 加载模型 checkpoint，提取 gate 值并分析专家利用率
使用 forward hook 捕获 gate 输出，不需要重新训练模型
"""
import sys, os, json, pickle
import numpy as np
import torch
from torch import nn
from collections import defaultdict

# Add paths
sys.path.insert(0, '/root/mtl4rec')
sys.path.insert(0, '/root/mtl4rec/fuxictr')

from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import H5DataLoader
import models as model_zoo

TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'is_hate', 'is_profile_enter']

def collect_gate_hooks(model):
    """注册 forward hook 收集所有 gate 层的输出"""
    gate_data = defaultdict(list)
    hooks = []
    
    def make_hook(layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 2:
                gate_data[layer_name].append(output.detach().cpu().numpy())
        return hook_fn
    
    for name, module in model.named_modules():
        # CGC gate layers (PLE, PLE_v14, PLE_v15): "cgc_layers.0.gate.0"
        if re.search(r'cgc_layers\.\d+\.gate\.\d+$', name) and 'gate_net' not in name:
            hook = module.register_forward_hook(make_hook(f"cgc_{name}"))
            hooks.append(hook)
        
        # PPNet gate nets (PLE_v13): "tower.X.gate_nets.Y"
        if 'gate_nets' in name and 'mlp' not in name:
            hook = module.register_forward_hook(make_hook(f"ppnet_{name}"))
            hooks.append(hook)
        
        # MMoE gate layers: "mmoe_layer.gate.0"
        if re.search(r'mmoe_layer\.gate\.\d+$', name):
            hook = module.register_forward_hook(make_hook(f"mmoe_{name}"))
            hooks.append(hook)
    
    return gate_data, hooks

import re

def compute_metrics(all_gate_data, num_experts):
    """从收集的 gate 数据计算专家利用率指标"""
    metrics = {}
    
    for layer_name, gate_list in all_gate_data.items():
        if len(gate_list) == 0:
            continue
        all_gates = np.concatenate(gate_list, axis=0)
        actual_experts = all_gates.shape[1]
        
        mean_gate = all_gates.mean(axis=0)
        eps = 1e-8
        entropy_per_sample = -np.sum(all_gates * np.log(all_gates + eps), axis=1)
        mean_entropy = entropy_per_sample.mean()
        max_entropy = np.log(actual_experts)
        normalized_entropy = mean_entropy / max_entropy if max_entropy > 0 else 0
        
        uniform_threshold = 1.0 / actual_experts
        utilized_experts = np.sum(mean_gate > uniform_threshold)
        utilization_rate = utilized_experts / actual_experts
        max_gate_per_sample = all_gates.max(axis=1)
        dominant_ratio = max_gate_per_sample.mean()
        usage_variance = np.var(mean_gate)
        
        metrics[layer_name] = {
            'mean_gate': mean_gate.tolist(),
            'mean_entropy': float(mean_entropy),
            'normalized_entropy': float(normalized_entropy),
            'utilization_rate': float(utilization_rate),
            'dominant_ratio': float(dominant_ratio),
            'usage_variance': float(usage_variance),
            'num_samples': int(all_gates.shape[0]),
            'num_experts': actual_experts
        }
    
    return metrics

def analyze_model(model_name, config_expid, config_path, checkpoint_path, max_batches=50):
    """分析单个模型的 gate 值"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")
    
    params = load_config(config_path, config_expid)
    params['gpu'] = -1
    
    # Fix data paths: H5DataLoader appends .h5, so strip .csv
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
    
    # Load checkpoint with strict=False to handle shape mismatches
    print(f"  Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    gate_data, hooks = collect_gate_hooks(model)
    
    params['batch_size'] = 4096
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    
    print(f"  Running inference on test data (max {max_batches} batches)...")
    batch_count = 0
    with torch.no_grad():
        for batch_data in test_gen:
            if batch_count >= max_batches:
                break
            try:
                model.forward(batch_data)
                batch_count += 1
            except Exception as e:
                print(f"  Warning: batch {batch_count} failed: {e}")
                batch_count += 1
                continue
    
    for h in hooks:
        h.remove()
    
    num_experts = params.get('num_shared_experts', 4) + params.get('num_specific_experts', 2)
    metrics = compute_metrics(gate_data, num_experts)
    
    print(f"  Captured {len(gate_data)} gate layers, {len(metrics)} with valid data")
    for name, m in sorted(metrics.items()):
        print(f"    {name}: entropy={m['normalized_entropy']:.3f}, "
              f"util={m['utilization_rate']:.2f}, "
              f"dominant={m['dominant_ratio']:.3f}, "
              f"experts={m['num_experts']}")
    
    return {
        'model_name': model_name,
        'gate_data': {k: [g.tolist() for g in v] for k, v in gate_data.items()},
        'metrics': metrics
    }

def main():
    base_dir = '/root/mtl4rec'
    config_path = os.path.join(base_dir, 'config')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints', 'KuaiRand1K')
    
    # 模型定义: (model_name, config_expid, checkpoint_file, config_version_note)
    models_to_analyze = [
        ('PLE_v5_2layer', 'PLE_KuaiRand1K', 'PLE_KuaiRand1K.model'),
        ('PLE_v13_PPNet', 'PLE_v13_KuaiRand1K', 'PLE_v13_KuaiRand1K.model'),
        ('PLE_v14_UserBias', 'PLE_v14_KuaiRand1K', 'PLE_v14_KuaiRand1K.model'),
        ('PLE_v15_SharedRes', 'PLE_v15_KuaiRand1K', 'PLE_v15_KuaiRand1K.model'),
        ('MMoE_v2_PPNet', 'MMoE_v2_KuaiRand1K', 'MMoE_v2_KuaiRand1K.model'),
    ]
    
    all_results = {}
    for model_name, expid, ckpt_file in models_to_analyze:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"SKIP: checkpoint not found: {ckpt_path}")
            continue
        
        try:
            result = analyze_model(model_name, expid, config_path, ckpt_path)
            all_results[model_name] = result
        except Exception as e:
            print(f"ERROR analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save metrics
    output_path = os.path.join(base_dir, 'results', 'gate_analysis.json')
    metrics_only = {k: v['metrics'] for k, v in all_results.items()}
    with open(output_path, 'w') as f:
        json.dump(metrics_only, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    
    # Save full gate data
    pickle_path = os.path.join(base_dir, 'results', 'gate_data.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Full gate data saved to {pickle_path}")

if __name__ == '__main__':
    main()

