#!/usr/bin/env python3
"""
任务相关性分析 — 用于面试解释模型表现
计算 8 个任务标签的 Pearson 相关系数，输出热力图和文本分析
"""
import pandas as pd
import numpy as np
import sys

TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment', 
         'is_forward', 'long_view', 'is_hate', 'is_profile_enter']
TASKS_CN = ['点击', '点赞', '关注', '评论', '转发', '长观看', '负反馈', '进主页']

# 训练集路径
TRAIN_PATH = '/root/mtl4rec/data/KuaiRand1K/train.csv'

def main():
    print("=" * 55)
    print("  8 任务标签 Pearson 相关性分析")
    print("=" * 55)
    
    # 读取样本 (全量可能太大，取 5 万行)
    print(f"\n[1/3] 读取训练集: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH, nrows=50000)
    print(f"      采样 {len(df)} 条记录")
    
    # 提取标签
    print(f"[2/3] 提取 8 个任务标签...")
    labels = {}
    for t in TASKS:
        if t in df.columns:
            labels[t] = df[t].values
            pos_rate = labels[t].mean()
            print(f"      {t:20s}  正样本率: {pos_rate:.4f} ({pos_rate*100:.2f}%)")
        else:
            print(f"      [WARN] {t} 不在训练集中!")
            sys.exit(1)
    
    # 计算 Pearson 相关系数矩阵
    print(f"\n[3/3] 计算 Pearson 相关系数矩阵...")
    label_matrix = np.column_stack([labels[t] for t in TASKS])
    corr = np.corrcoef(label_matrix.T)
    
    # ===== 输出 1: 完整矩阵 =====
    print("\n" + "=" * 55)
    print("  完整 Pearson 相关系数矩阵")
    print("=" * 55)
    print(f"{'':>15}", end="")
    for cn in TASKS_CN:
        print(f"{cn:>6}", end="")
    print()
    for i, t in enumerate(TASKS):
        print(f"{TASKS_CN[i]:>12}  ", end="")
        for j in range(len(TASKS)):
            if i == j:
                print(f"  --  ", end="")
            else:
                print(f"{corr[i][j]:6.3f}", end="")
        print()
    
    # ===== 输出 2: 排序后的任务对 =====
    print("\n" + "=" * 55)
    print("  任务对相关性排名 (Top 10)")
    print("=" * 55)
    pairs = []
    for i in range(len(TASKS)):
        for j in range(i+1, len(TASKS)):
            pairs.append((TASKS_CN[i], TASKS_CN[j], corr[i][j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for rank, (a, b, r) in enumerate(pairs[:10], 1):
        bar = "█" * int(abs(r) * 40)
        direction = "正相关" if r > 0 else "负相关"
        print(f"  {rank:2d}. {a:6s} ↔ {b:6s}  r={r:+.3f}  {bar} {direction}")
    
    # ===== 输出 3: 面试话术 =====
    print("\n" + "=" * 55)
    print("  面试分析话术")
    print("=" * 55)
    
    # 找最强正相关
    max_pair = max(pairs, key=lambda x: x[2])
    print(f"\n★ 最强正相关: {max_pair[0]} ↔ {max_pair[1]} (r={max_pair[2]:.3f})")
    print(f"  → 这两个任务共享底层表示是有利的，SharedBottom 在这些任务上不会差")
    
    # 统计弱相关比例
    weak_count = sum(1 for _, _, r in pairs if abs(r) < 0.1)
    print(f"\n★ 弱相关任务对: {weak_count}/{len(pairs)} (r < 0.1)")
    print(f"  → 大多数任务之间几乎独立，这正是 MMoE/PLE 应该发挥优势的场景")
    print(f"  → 如果 PLE 没赢，说明它的门控机制没有充分学习到任务差异")
    
    # 按任务统计平均相关性
    print(f"\n★ 各任务与其他任务的平均 |r|:")
    task_avg_corr = {}
    for i, t in enumerate(TASKS):
        avg = np.mean([abs(corr[i][j]) for j in range(len(TASKS)) if j != i])
        task_avg_corr[TASKS_CN[i]] = avg
    
    for name, avg in sorted(task_avg_corr.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(avg * 200)
        print(f"  {name:6s}: avg|r|={avg:.3f}  {bar}")
    
    print(f"\n  → 相关性越高的任务，SharedBottom 表现越好")
    print(f"  → 相关性越低的任务，越需要 MMoE/PLE 的门控机制来分离")
    
    # ===== 输出 4: 保存 JSON 供可视化用 =====
    import json
    result = {
        "tasks": TASKS,
        "tasks_cn": TASKS_CN,
        "correlation": corr.tolist(),
        "pairs": [(a, b, float(r)) for a, b, r in pairs]
    }
    with open('/root/mtl4rec/task_correlation.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果已保存到 /root/mtl4rec/task_correlation.json")

if __name__ == '__main__':
    main()