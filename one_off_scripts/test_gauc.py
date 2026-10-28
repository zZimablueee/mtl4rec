
import sys, json
sys.path.insert(0, 'fuxictr')
from fuxictr.features import FeatureMap

fm = FeatureMap('KuaiRand1K', 'data/KuaiRand1K')
fm.load('data/KuaiRand1K/feature_map.json', {'dataset_id': 'KuaiRand1K', 'embedding_dim': 16})
fm.labels = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'is_hate', 'is_profile_enter']
fm.set_column_index()
print('group_id:', fm.group_id)
print('user_id in column_index:', 'user_id' in fm.column_index)
print('input_length:', fm.input_length)
# Search for user_id in features
for feat in fm.features:
    if 'user_id' in feat:
        print('FOUND user_id in features:', feat)
        break
else:
    print('user_id NOT in features')
