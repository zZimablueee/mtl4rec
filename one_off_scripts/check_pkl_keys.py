import pickle
for path in ['/root/mtl4rec/results/predictions/SharedBottom_base_predictions.pkl',
             '/root/mtl4rec/results/predictions/MMoE_v3_predictions.pkl',
             '/root/mtl4rec/results/predictions/PLE_v7_predictions.pkl']:
    with open(path, 'rb') as f:
        pred = pickle.load(f)
    print(f"=== {path.split('/')[-1]} ===")
    print(f"type: {type(pred)}")
    if isinstance(pred, dict):
        print(f"keys: {list(pred.keys())}")
        for k in list(pred.keys())[:3]:
            v = pred[k]
            shape = v.shape if hasattr(v, 'shape') else f'len={len(v)}'
            print(f"  {k}: {type(v).__name__}, {shape}")
    print()