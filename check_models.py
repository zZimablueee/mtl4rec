import json
d = json.load(open('/root/mtl4rec/results/analysis_results.json'))
for k, v in sorted(d.items(), key=lambda x: x[1]['test_auc']['avg_auc'], reverse=True):
    print(f"{k:<30} avg={v['test_auc']['avg_auc']:.4f}  forward={v['test_auc']['is_forward']:.4f}  comment={v['test_auc']['is_comment']:.4f}")