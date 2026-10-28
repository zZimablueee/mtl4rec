import pickle, numpy as np
from sklearn.metrics import roc_auc_score

TASKS = ['is_click', 'is_like', 'is_follow', 'is_comment',
         'is_forward', 'long_view', 'is_hate', 'is_profile_enter']

models = {
    'SharedBottom': '/root/mtl4rec/results/predictions/SharedBottom_base_predictions.pkl',
    'MMoE_v3':     '/root/mtl4rec/results/predictions/MMoE_v3_predictions.pkl',
    'PLE_v7':      '/root/mtl4rec/results/predictions/PLE_v7_predictions.pkl',
}

for name, path in models.items():
    print(f"\n===== {name} =====")
    with open(path, 'rb') as f:
        pred = pickle.load(f)
    
    group_ids = pred['group_ids']
    unique_groups = np.unique(group_ids)
    print(f"n_samples={pred['n_samples']}, n_groups={len(unique_groups)}")
    
    print(f"{'Task':<20s} {'AUC':>8s}  {'gAUC':>8s}  {'diff':>8s}")
    print('-' * 52)
    
    avg_auc = 0
    avg_gauc = 0
    for task in TASKS:
        y_true = pred['y_true'][task]
        y_pred = pred['y_pred'][task]
        
        if len(np.unique(y_true)) < 2:
            auc = float('nan')
        else:
            auc = roc_auc_score(y_true, y_pred)
        
        gauc_num = 0.0
        gauc_den = 0.0
        for g in unique_groups:
            mask = group_ids == g
            yt = y_true[mask]
            yp = y_pred[mask]
            if len(np.unique(yt)) < 2:
                continue
            n = len(yt)
            try:
                g_auc = roc_auc_score(yt, yp)
            except:
                continue
            gauc_num += g_auc * n
            gauc_den += n
        
        gauc = gauc_num / gauc_den if gauc_den > 0 else float('nan')
        
        print(f"{task:<20s} {auc:8.4f}  {gauc:8.4f}  {auc-gauc:+8.4f}")
        avg_auc += auc
        avg_gauc += gauc
    
    avg_auc /= len(TASKS)
    avg_gauc /= len(TASKS)
    print(f"{'avg':<20s} {avg_auc:8.4f}  {avg_gauc:8.4f}  {avg_auc-avg_gauc:+8.4f}")
