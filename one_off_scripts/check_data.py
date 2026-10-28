import pandas as pd

tr = pd.read_csv('data/KuaiRand1K/train.csv')
te = pd.read_csv('data/KuaiRand1K/test.csv')
va = pd.read_csv('data/KuaiRand1K/valid.csv')

print('=== 数据集规模 ===')
print(f'训练集: {len(tr)} 样本')
print(f'验证集: {len(va)} 样本')
print(f'测试集: {len(te)} 样本')
print(f'总样本: {len(tr)+len(va)+len(te)}')
print()

print('=== 独立用户/视频 ===')
print(f'训练集用户: {tr.user_id.nunique()}, 视频: {tr.video_id.nunique()}')
print(f'验证集用户: {va.user_id.nunique()}, 视频: {va.video_id.nunique()}')
print(f'测试集用户: {te.user_id.nunique()}, 视频: {te.video_id.nunique()}')
print()

labels = ['is_click','is_like','is_follow','is_comment','is_forward','long_view','is_hate','is_profile_enter']
print('=== 训练集标签正样本率 ===')
for l in labels:
    p = tr[l].mean()
    n = tr[l].sum()
    print(f'  {l}: {p:.4f} ({p*100:.2f}%)  正样本数={int(n)}')
print()

print('=== 测试集标签正样本率 ===')
for l in labels:
    p = te[l].mean()
    n = te[l].sum()
    print(f'  {l}: {p:.4f} ({p*100:.2f}%)  正样本数={int(n)}')
print()

excluded = ['play_time_ms', 'profile_stay_time', 'comment_stay_time']
kept = [c for c in tr.columns if c not in excluded]
print('=== 列信息 ===')
print(f'原始列数: {len(tr.columns)}')
print(f'排除了: {excluded}')
print(f'保留列数: {len(kept)}')
print()

print('=== 保留的所有列 ===')
print(kept)
print()

print('=== video_duration vs duration_ms vs play_duration ===')
print(f'duration_ms mean: {tr.duration_ms.mean():.1f}')
print(f'video_duration mean: {tr.video_duration.mean():.1f}')
print(f'play_duration mean: {tr.play_duration.mean():.1f}')
print(f'video_duration == duration_ms: {(tr.duration_ms == tr.video_duration).mean():.4f}')
