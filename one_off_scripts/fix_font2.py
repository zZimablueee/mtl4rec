#!/usr/bin/env python3
"""Fix all Chinese titles in full_analysis.py to English"""
import re

with open('/root/mtl4rec/full_analysis.py', 'r') as f:
    content = f.read()

# Replace all Chinese titles with English
replacements = [
    ("模型 Test AUC 对比 (KuaiRand-1K)", "Model Test AUC Comparison (KuaiRand-1K)"),
    ("任务级 AUC 热力图 (KuaiRand-1K)", "Task-level AUC Heatmap (KuaiRand-1K)"),
    ("各模型相对", "Task AUC Improvement vs "),
    (" 的任务提升幅度", ""),
    ("AUC 变化 (vs", "AUC Change (vs"),
    ("Gate / 专家利用率分析", "Gate / Expert Utilization Analysis"),
    ("Gate 归一化熵", "Gate Normalized Entropy"),
    ("专家利用率", "Expert Utilization Rate"),
    ("Gate 主导比率", "Gate Dominant Ratio"),
    ("KuaiRand-1K 多任务学习实验综合分析", "KuaiRand-1K Multi-Task Learning Analysis"),
    ("Top 10 模型 Avg AUC 排名", "Top 10 Model Avg AUC Ranking"),
    ("任务相关性", "Task Correlation"),
    ("关键模型任务级 AUC 对比", "Key Model Task-level AUC Comparison"),
    ("Gate 熵 vs 模型性能", "Gate Entropy vs Model Performance"),
    ("所有模型 Test AUC 完整对比", "All Models Test AUC Full Comparison"),
    ("任务相关性 (N/A)", "Task Correlation (N/A)"),
    ("Correlation data\\nunavailable", "Correlation data\\nunavailable"),
    ("Gate data\\nunavailable", "Gate data\\nunavailable"),
    ("Best:", "Best:"),
]

for old, new in replacements:
    content = content.replace(old, new)

# Also update key_models in plot_model_comparison and plot_task_heatmap
# Add PLE_v5, PLE_v13, PLE_v14 to the key_models lists
old_key = """key_models = ['SharedBottom_base', 'MMoE_base', 'PLE_base',
                  'PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14',
                  'MMoE_v2']"""
new_key = """key_models = ['PLE_v5', 'SharedBottom_base', 'MMoE_base', 'PLE_base',
                  'PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14',
                  'MMoE_v2']"""
content = content.replace(old_key, new_key)

# Also update the key_models in main
old_main_key = """key_models = ['PLE_base', 'PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14', 'MMoE_v2', 'MMoE_base', 'SharedBottom_base']"""
new_main_key = """key_models = ['PLE_v5', 'PLE_base', 'PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14', 'MMoE_v2', 'MMoE_base', 'SharedBottom_base']"""
content = content.replace(old_main_key, new_main_key)

with open('/root/mtl4rec/full_analysis.py', 'w') as f:
    f.write(content)

print("All Chinese titles replaced with English")
print(f"File size: {len(content)} bytes")