"""Fix full_analysis.py key_models to include all important models"""
import re

with open('/root/mtl4rec/full_analysis.py', 'r') as f:
    content = f.read()

# The correct key_models list (all 3 places it appears)
correct_key_models = """['PLE_v5', 'PLE_v7', 'PLE_v14', 'MMoE_v3', 'PLE_v3',
                  'SharedBottom_v4', 'PLE_v13', 'SharedBottom_base',
                  'MMoE_v2', 'PLE_v12', 'PLE_v6', 'PLE_base', 'MMoE_base']"""

# Replace in plot_model_comparison
old1 = """key_models = ['PLE_v5', 'SharedBottom_base', 'MMoE_base', 'PLE_base',
                  'PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14',
                  'MMoE_v2']"""
content = content.replace(old1, "key_models = " + correct_key_models)

# Replace in plot_task_heatmap  
old2 = """key_models = ['PLE_v5', 'SharedBottom_base', 'MMoE_base', 'PLE_base',
                  'PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14',
                  'MMoE_v2']"""
content = content.replace(old2, "key_models = " + correct_key_models)

# Replace in main() key_models
old3 = """key_models = ['PLE_v5', 'PLE_base', 'PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14', 'MMoE_v2', 'MMoE_base', 'SharedBottom_base']"""
content = content.replace(old3, "key_models = " + correct_key_models + " + ['SharedBottom_v3']")

# Also update compare_keys in plot_task_improvement and plot_summary_dashboard
old4 = """compare_keys = ['PLE_base', 'PLE_v6', 'PLE_v7', 'PLE_v13', 'PLE_v14', 'MMoE_v2']"""
new4 = """compare_keys = ['PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14', 'MMoE_v2', 'MMoE_v3']"""
content = content.replace(old4, new4)

with open('/root/mtl4rec/full_analysis.py', 'w') as f:
    f.write(content)

print("full_analysis.py key_models updated")
print(f"File size: {len(content)} bytes")