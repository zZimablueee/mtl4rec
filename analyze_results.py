
#!/usr/bin/env python3
"""
analyze_results.py - 从所有训练日志中提取 test AUC 结果并进行对比分析
不需要重新训练模型，直接解析已有日志文件
"""
import os, re, json
from pathlib import Path
from collections import defaultdict
import numpy as np

LOG_DIR = "./results/logs"
TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'is_hate', 'is_profile_enter']

def parse_log(filepath):
    """解析单个训练日志，提取 test AUC 值"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 提取 test 阶段的 AUC
    test_auc = {}
    test_lines = content.split("Test evaluation")
    if len(test_lines) < 2:
        # 有些日志可能没有 test evaluation（比如中断的）
        return None
    
    test_section = test_lines[-1]  # 取最后一个 test evaluation
    for task in TASKS:
        pattern = rf'\[Task: {task}\].*?AUC:\s*([\d.]+)'
        match = re.search(pattern, test_section)
        if match:
            test_auc[task] = float(match.group(1))
    
    if len(test_auc) < 8:
        return None
    
    test_auc['avg_auc'] = np.mean(list(test_auc.values()))
    
    # 提取验证集 AUC（选最佳 epoch）
    val_auc = {}
    val_blocks = content.split("****** Validation evaluation ******")
    if len(val_blocks) > 1:
        val_section = val_blocks[-1]
        for task in TASKS:
            pattern = rf'\[Task: {task}\].*?AUC:\s*([\d.]+)'
            match = re.search(pattern, val_section)
            if match:
                val_auc[task] = float(match.group(1))
        if len(val_auc) == 8:
            val_auc['avg_auc'] = np.mean(list(val_auc.values()))
    
    return {
        'test_auc': test_auc,
        'val_auc': val_auc if len(val_auc) == 8 else None
    }

def identify_model(log_name):
    """从日志文件名中识别模型名称和版本"""
    name = log_name.replace('.log', '')
    # 匹配模式: {Model}_KuaiRand1K_{version}_{timestamp}
    # 例如: PLE_KuaiRand1K_v12_20260719_202428
    # 或: PLE_KuaiRand1K_20260709_224501 (无版本号)
    
    # 先尝试匹配有版本号的
    m = re.match(r'(PLE_v?\d+|MMoE_v?\d+|SharedBottom|PLE|MMoE)_KuaiRand1K_(v?\d+[a-z]*)?_?(\d{8}_\d{6})', name)
    if m:
        model_name = m.group(1)
        version = m.group(2) if m.group(2) else 'base'
        return model_name, version
    
    # 匹配无版本号的基础模型
    m = re.match(r'(SharedBottom|MMoE|PLE)_KuaiRand1K_(\d{8}_\d{6})', name)
    if m:
        return m.group(1), 'base'
    
    return name, 'unknown'

def main():
    results = {}
    log_dir = Path(LOG_DIR)
    
    for log_file in sorted(log_dir.glob('*.log')):
        model_name, version = identify_model(log_file.name)
        data = parse_log(str(log_file))
        if data is None:
            continue
        
        key = f"{model_name}_{version}"
        # 如果同一个模型有多个日志，取 avg_auc 最高的
        if key not in results or data['test_auc']['avg_auc'] > results[key]['test_auc']['avg_auc']:
            results[key] = {
                'model': model_name,
                'version': version,
                'log_file': log_file.name,
                **data
            }
    
    # 排序输出
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_auc']['avg_auc'], reverse=True)
    
    print("=" * 120)
    print(f"{'Model':<25} {'Version':<10} {'click':<8} {'like':<8} {'follow':<8} {'comment':<8} {'forward':<8} {'long_view':<8} {'hate':<8} {'profile':<8} {'avg_auc':<8}")
    print("-" * 120)
    
    for key, info in sorted_results:
        auc = info['test_auc']
        print(f"{info['model']:<25} {info['version']:<10} "
              f"{auc['is_click']:.4f}   {auc['is_like']:.4f}   {auc['is_follow']:.4f}   "
              f"{auc['is_comment']:.4f}   {auc['is_forward']:.4f}   {auc['long_view']:.4f}   "
              f"{auc['is_hate']:.4f}   {auc['is_profile_enter']:.4f}   "
              f"{auc['avg_auc']:.4f}")
    
    print("-" * 120)
    
    # 保存 JSON
    output = {k: {
        'model': v['model'], 'version': v['version'],
        'log_file': v['log_file'],
        'test_auc': {t: round(v['test_auc'][t], 6) for t in TASKS + ['avg_auc']},
        'val_auc': {t: round(v['val_auc'][t], 6) for t in TASKS + ['avg_auc']} if v['val_auc'] else None
    } for k, v in sorted_results}
    
    with open('results/analysis_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to results/analysis_results.json ({len(output)} models)")
    
    # 打印最佳模型对比
    print("\n" + "=" * 80)
    print("关键模型对比 (Test Avg AUC)")
    print("=" * 80)
    
    key_models = ['SharedBottom_base', 'MMoE_base', 'PLE_base', 
                  'PLE_v6', 'PLE_v7', 'PLE_v12', 'PLE_v13', 'PLE_v14', 'PLE_v15',
                  'MMoE_v2']
    for key in key_models:
        if key in results:
            info = results[key]
            print(f"  {key:<25} avg_auc={info['test_auc']['avg_auc']:.4f}  "
                  f"forward={info['test_auc']['is_forward']:.4f}  "
                  f"comment={info['test_auc']['is_comment']:.4f}")

if __name__ == '__main__':
    main()

