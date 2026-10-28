
import pickle, numpy as np

# Check actual group_ids in predictions
for name in ['SharedBottom_base', 'MMoE_v3', 'PLE_v7']:
    path = f'/root/mtl4rec/results/predictions/{name}_predictions.pkl'
    with open(path, 'rb') as f:
        pred = pickle.load(f)
    gids = pred['group_ids']
    print(f"{name}: group_ids type={type(gids)}, is None={gids is None}")
    if gids is not None:
        print(f"  shape={gids.shape}, unique={len(np.unique(gids))}")
        print(f"  first 5: {gids[:5]}")
    else:
        print(f"  group_ids is None - no user_id in predictions")
    print()
