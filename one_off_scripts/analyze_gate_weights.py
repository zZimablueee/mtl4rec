
import sys, os, json
sys.path.insert(0, '/root/mtl4rec')
sys.path.insert(0, '/root/mtl4rec/fuxictr')
import torch
import numpy as np

from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
import models as model_zoo

base_dir = '/root/mtl4rec'
config_path = os.path.join(base_dir, 'config')
TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'is_hate', 'is_profile_enter']

def analyze_gate_weights(model_name, expid, ckpt_file):
    print(f"\n{'='*60}")
    print(f"Gate Weight Analysis: {model_name}")
    print(f"{'='*60}")
    
    params = load_config(config_path, expid)
    params['gpu'] = -1
    
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    
    task_labels = params['task_labels'] if isinstance(params['task_labels'], list) else list(params['task_labels'])
    feature_map.labels = task_labels
    feature_map.set_column_index()
    
    model = getattr(model_zoo, params['model'])(feature_map, **params)
    ckpt = torch.load(os.path.join(base_dir, 'checkpoints/KuaiRand1K', ckpt_file), map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    # For CGC models, analyze gate weights
    if hasattr(model, 'cgc_layers'):
        for layer_idx, cgc_layer in enumerate(model.cgc_layers):
            print(f"\n  CGC Layer {layer_idx}:")
            gate_norms = []
            for i in range(len(cgc_layer.gate)):
                # Get first linear layer weight
                gate_w = cgc_layer.gate[i].mlp[0].weight
                # L2 norm of each output neuron
                w_norm = gate_w.norm(dim=1).detach().cpu().numpy()
                gate_norms.append(w_norm)
            
            # Print as table
            print(f"    {'Task':<20} {'L2 Norm':<30} {'Mean':<10}")
            for i in range(min(8, len(gate_norms))):
                task_name = TASKS[i] if i < 8 else f"shared_{i}"
                print(f"    {task_name:<20} {str(gate_norms[i][:5]):<30} {gate_norms[i].mean():.4f}")
    
    # For MMoE models
    if hasattr(model, 'mmoe_layer'):
        print(f"\n  MMoE Gate Layer:")
        gate_norms = []
        for i in range(len(model.mmoe_layer.gate)):
            gate_w = model.mmoe_layer.gate[i].mlp[0].weight
            w_norm = gate_w.norm(dim=1).detach().cpu().numpy()
            gate_norms.append(w_norm)
        
        print(f"    {'Task':<20} {'L2 Norm':<30} {'Mean':<10}")
        for i in range(len(gate_norms)):
            print(f"    {TASKS[i]:<20} {str(gate_norms[i][:8]):<30} {gate_norms[i].mean():.4f}")

# Analyze models
analyze_gate_weights('PLE_v5', 'PLE_KuaiRand1K', 'PLE_KuaiRand1K.model')
analyze_gate_weights('PLE_v14', 'PLE_v14_KuaiRand1K', 'PLE_v14_KuaiRand1K.model')
analyze_gate_weights('PLE_v15', 'PLE_v15_KuaiRand1K', 'PLE_v15_KuaiRand1K.model')
analyze_gate_weights('MMoE_v2', 'MMoE_v2_KuaiRand1K', 'MMoE_v2_KuaiRand1K.model')

