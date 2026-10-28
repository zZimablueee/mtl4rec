import csv

labels = ['is_click','is_like','is_follow','is_comment','is_forward','long_view','is_hate','is_profile_enter']

for split, path in [('train', 'data/KuaiRand1K/train.csv'), 
                     ('valid', 'data/KuaiRand1K/valid.csv'),
                     ('test', 'data/KuaiRand1K/test.csv')]:
    print(f'\n=== {split} ===')
    counts = {l: 0 for l in labels}
    total = 0
    users = set()
    videos = set()
    dm_diff = 0
    dm_total = 0
    
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        label_indices = [header.index(l) for l in labels]
        u_idx = header.index('user_id')
        v_idx = header.index('video_id')
        dm_idx = header.index('duration_ms')
        vd_idx = header.index('video_duration')
        
        for row in reader:
            total += 1
            users.add(row[u_idx])
            videos.add(row[v_idx])
            for i, l in zip(label_indices, labels):
                val = row[i].strip()
                if val:
                    counts[l] += int(float(val))
            dm_val = row[dm_idx].strip()
            vd_val = row[vd_idx].strip()
            if dm_val and vd_val:
                dm_total += 1
                if abs(float(dm_val) - float(vd_val)) < 0.01:
                    dm_diff += 1
    
    print(f'样本数: {total}')
    print(f'独立用户: {len(users)}')
    print(f'独立视频: {len(videos)}')
    print('标签正样本率:')
    for l in labels:
        p = counts[l] / total
        print(f'  {l}: {p:.4f} ({p*100:.2f}%)  正样本={counts[l]}')
    if dm_total > 0:
        print(f'duration_ms == video_duration: {dm_diff}/{dm_total} = {dm_diff/dm_total:.4f}')
