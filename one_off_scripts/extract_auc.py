import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
import json

TASKS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward",
         "long_view", "is_hate", "is_profile_enter"]

single_auc = {
    "is_click": 0.7144, "is_like": 0.7523, "is_follow": 0.6596,
    "is_comment": 0.7143, "is_forward": 0.6307, "long_view": 0.7327,
    "is_hate": 0.6394, "is_profile_enter": 0.6375
}

MODELS = {
    "SharedBottom": "/root/mtl4rec/results/predictions/SharedBottom_base_predictions.pkl",
    "MMoE_v3": "/root/mtl4rec/results/predictions/MMoE_v3_predictions.pkl",
    "PLE_v7": "/root/mtl4rec/results/predictions/PLE_v7_predictions.pkl",
}

result = {"single": {}, "mtl": {}}
result["single"] = {t: {"auc": single_auc[t]} for t in TASKS}

for model_name, pred_path in MODELS.items():
    with open(pred_path, "rb") as f:
        pred = pickle.load(f)
    result["mtl"][model_name] = {}
    for t in TASKS:
        yt = pred["y_true"][t]
        yp = pred["y_pred"][t]
        auc = roc_auc_score(yt, yp)
        pos_rate = yt.mean() * 100
        result["mtl"][model_name][t] = {"auc": round(auc, 4), "pos_rate": round(pos_rate, 2)}
    avg = np.mean([result["mtl"][model_name][t]["auc"] for t in TASKS])
    result["mtl"][model_name]["avg_auc"] = round(avg, 4)

print(json.dumps(result, indent=2))
