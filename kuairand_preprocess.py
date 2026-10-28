#!/usr/bin/env python
"""KuaiRand-1K 数据预处理 v3 — 修正列名"""
import pandas as pd
import numpy as np
import os, argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/root/autodl-fs/KuaiRand-1K/data')
    p.add_argument('--output_dir', default='./data/KuaiRand1K')
    p.add_argument('--valid_ratio', type=float, default=0.1)
    p.add_argument('--test_ratio', type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载用户特征
    df_user = pd.read_csv(os.path.join(args.data_dir, 'user_features_1k.csv'))
    for col in df_user.columns:
        if df_user[col].dtype in [np.int64, np.float64]:
            df_user[col] = df_user[col].replace(-124, np.nan)
    df_user['is_live_streamer'] = df_user['is_live_streamer'].fillna(0).astype(int)
    df_user['is_video_author'] = df_user['is_video_author'].fillna(0).astype(int)
    df_user['user_active_degree'] = df_user['user_active_degree'].astype(str)
    for col in ['follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days']:
        if col in df_user.columns:
            df_user[col] = df_user[col].fillna(0).astype(float)

    # 加载视频特征
    df_video_basic = pd.read_csv(os.path.join(args.data_dir, 'video_features_basic_1k.csv'))
    df_video_stat = pd.read_csv(os.path.join(args.data_dir, 'video_features_statistic_1k.csv'))
    df_video = df_video_basic.merge(df_video_stat, on='video_id', how='left')

    # 加载交互日志
    df = pd.read_csv(os.path.join(args.data_dir, 'log_standard_4_22_to_5_08_1k.csv'))
    print(f"Total interactions: {len(df):,}")

    # 合并
    df = df.merge(df_user, on='user_id', how='left')
    df = df.merge(df_video, on='video_id', how='left')
    print(f"After merging: {len(df):,} rows, {len(df.columns)} cols")

    # === 分类特征 ===
    cat_features = [
        'user_id', 'video_id', 'author_id', 'tab', 'video_type',
        'upload_type', 'user_active_degree', 'is_live_streamer', 'is_video_author',
        'visible_status', 'music_id', 'music_type', 'is_lowactive_period',
        'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range',
        'register_days_range',
    ]

    # === 数值特征 ===
    num_features = [
        # 交互日志里的
        'play_time_ms', 'duration_ms', 'profile_stay_time', 'comment_stay_time',
        # 用户特征
        'follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days',
        # 视频基础
        'video_duration', 'server_width', 'server_height',
        # 视频统计 (选关键指标)
        'show_cnt', 'play_cnt', 'play_user_num', 'play_duration',
        'complete_play_cnt', 'valid_play_cnt', 'long_time_play_cnt',
        'play_progress', 'like_cnt', 'like_user_num', 'comment_cnt',
        'comment_user_num', 'follow_cnt', 'share_cnt', 'collect_cnt',
        'download_cnt', 'report_cnt',
        # 18个加密特征
        'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
        'onehot_feat4', 'onehot_feat5', 'onehot_feat6', 'onehot_feat7',
        'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11',
        'onehot_feat12', 'onehot_feat13', 'onehot_feat14', 'onehot_feat15',
        'onehot_feat16', 'onehot_feat17',
    ]

    # 标签
    label_cols = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
              'long_view', 'is_hate', 'is_profile_enter']

    keep_cols = [c for c in cat_features + num_features if c in df.columns] + label_cols
    df = df[keep_cols].copy()
    print(f"Final: {len(df):,} rows, {len(keep_cols)} cols")
    print(f"  Cat features: {len([c for c in cat_features if c in df.columns])}")
    print(f"  Num features: {len([c for c in num_features if c in df.columns])}")

    total = len(df)
    test_n = int(total * args.test_ratio)
    valid_n = int(total * args.valid_ratio)
    train_n = total - valid_n - test_n

    train = df.iloc[:train_n].reset_index(drop=True)
    valid = df.iloc[train_n:train_n+valid_n].reset_index(drop=True)
    test = df.iloc[train_n+valid_n:].reset_index(drop=True)

    print(f"Train: {len(train):,} | Valid: {len(valid):,} | Test: {len(test):,}")
    train.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    valid.to_csv(os.path.join(args.output_dir, 'valid.csv'), index=False)
    test.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    print("Done!")

if __name__ == '__main__':
    main()
