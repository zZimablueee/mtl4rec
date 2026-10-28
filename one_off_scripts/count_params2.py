import sys, os
sys.path.insert(0, "/root/STEM/fuxictr")
os.chdir("/root/mtl4rec")
import torch
import yaml

# Load config
with open("config/model_config_v3.yaml") as f:
    config = yaml.safe_load(f)

# Build a minimal feature_map
from fuxictr import FeatureMap
import pandas as pd
import numpy as np

# Read data to get feature info
from fuxictr.pytorch.datasets import Dataset
from fuxictr.pytorch.dataloaders import DataLoader

# Just create a dummy feature map
feature_cols = [
    "user_id", "video_id", "author_id", "tab", "video_type", "upload_type",
    "user_active_degree", "is_live_streamer", "is_video_author", "visible_status",
    "music_id", "music_type", "is_lowactive_period",
    "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range"
]
num_cols = [
    "duration_ms", "fans_user_num", "friend_user_num", "register_days",
    "video_duration", "server_width", "server_height",
    "show_cnt", "play_cnt", "play_user_num", "play_duration", "complete_play_cnt",
    "valid_play_cnt", "long_time_play_cnt", "play_progress",
    "like_cnt", "like_user_num", "comment_cnt", "comment_user_num",
    "follow_cnt", "share_cnt", "collect_cnt", "download_cnt", "report_cnt"
] + [f"onehot_feat{i}" for i in range(18)]

# Read train data
df = pd.read_csv("data/KuaiRand1K/train.csv", nrows=100)
feature_map = FeatureMap(dataset_id="KuaiRand1K")
# Manually set feature map
for col in feature_cols:
    feature_map.features[col] = {"type": "categorical", "vocab_size": 100}
for col in num_cols:
    feature_map.features[col] = {"type": "numeric"}
feature_map.num_fields = len(feature_cols) + len(num_cols)
feature_map.labels = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view", "is_hate", "is_profile_enter"]

# Count SharedBottom
from models.SharedBottom import SharedBottom
sb_cfg = config["SharedBottom_KuaiRand1K"]
model_sb = SharedBottom(feature_map, **sb_cfg)
sb_params = sum(p.numel() for p in model_sb.parameters())
print(f"SharedBottom: {sb_params:,}")

# Count MMoE
from models.MMoE import MMoE
mmoe_cfg = config["MMoE_KuaiRand1K"]
model_mmoe = MMoE(feature_map, **mmoe_cfg)
mmoe_params = sum(p.numel() for p in model_mmoe.parameters())
print(f"MMoE: {mmoe_params:,}")

# Count PLE
from models.PLE import PLE
ple_cfg = config["PLE_KuaiRand1K"]
model_ple = PLE(feature_map, **ple_cfg)
ple_params = sum(p.numel() for p in model_ple.parameters())
print(f"PLE: {ple_params:,}")
