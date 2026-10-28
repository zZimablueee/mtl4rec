#!/bin/bash
cd /root/mtl4rec

# ===== V5: PLE 2层 CGC =====
cp config/model_config_v5.yaml config/model_config.yaml
/root/miniconda3/bin/python run_with_log.py PLE_KuaiRand1K ./config 0 v5

# ===== V6: PLE 2层 + Balance Loss =====
cp config/model_config_v6.yaml config/model_config.yaml
/root/miniconda3/bin/python run_with_log.py PLE_KuaiRand1K ./config 0 v6

# ===== V7: PLE 2层 + Uncertainty Weighting =====
cp config/model_config_v7.yaml config/model_config.yaml
/root/miniconda3/bin/python run_with_log.py PLE_KuaiRand1K ./config 0 v7
