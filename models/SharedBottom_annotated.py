# ============================================================
# SharedBottom.py — 共享底层多任务模型
# ============================================================
# 架构：Embedding → 共享底层MLP → 8个独立Tower → 8个预测概率
# 这是最基础的多任务学习架构，所有任务共享同一个底层网络
#
# 关键组件：
#   self.embedding_layer: 特征Embedding层，将离散特征映射为稠密向量
#   self.bottom: 共享底层MLP，所有8个任务共同使用的特征提取器
#   self.tower: 8个独立的Tower MLP，每个任务有自己的输出头
#   self.output_activation: 8个Sigmoid，将logits转为[0,1]概率
#
# 数据流（forward）：
#   输入特征 → Embedding → flatten → bottom MLP → 8个Tower → 8个Sigmoid → 8个预测
#
# 配置参数（model_config_v3.yaml）：
#   bottom_hidden_units: [512, 512, 512] — 3层共享底层，每层512维
#   tower_hidden_units: [128, 128]       — 每个任务Tower 2层，每层128维
#   embedding_dim: 16                     — 特征Embedding维度
#   loss_weight: [1.0, 2.0, 3.0, ...]   — 8个任务的损失权重
# ============================================================

import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block

class SharedBottom(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 model_id="SharedBottom",
                 gpu=-1,
                 task=["binary_classification"],
                 num_tasks=1,
                 loss_weight='EQ',
                 learning_rate=1e-3,
                 embedding_dim=10,
                 bottom_hidden_units=[64, 64, 64],
                 tower_hidden_units=[64, ],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(SharedBottom, self).__init__(feature_map,
                                           task=task,
                                           loss_weight=loss_weight,
                                           num_tasks=num_tasks,
                                           model_id=model_id,
                                           gpu=gpu,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        # ---- 1. Embedding 层 ----
        # 将所有离散特征（user_id, video_id, tab等）映射为 embedding_dim 维向量
        # 输出维度 = embedding_dim × feature_map.num_fields（所有特征域拼接）
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # ---- 2. 共享底层 (Bottom MLP) ----
        # 所有8个任务共享的全连接网络
        # 输入: embedding_dim × num_fields 维向量（如 16 × 50 = 800 维）
        # 输出: bottom_hidden_units[-1] 维向量（如 512 维）
        # 配置: 3层，每层512维，ReLU激活，BatchNorm，Dropout 0.1
        self.bottom = MLP_Block(input_dim=embedding_dim * feature_map.num_fields,
                                hidden_units=bottom_hidden_units,
                                hidden_activations=hidden_activations,
                                output_activation=None,  # 底层不做非线性输出
                                dropout_rates=net_dropout,
                                batch_norm=batch_norm)

        # ---- 3. 任务专属 Tower ----
        # 每个任务一个独立的MLP，从共享底层拿到表示后各自处理
        # 输入: bottom_hidden_units[-1] 维（如 512 维）
        # 输出: 1 维（logit，经 Sigmoid 后变为概率）
        # 配置: 2层，每层128维
        self.tower = nn.ModuleList([MLP_Block(input_dim=bottom_hidden_units[-1],
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=hidden_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout,
                                              batch_norm=batch_norm)
                                    for _ in range(num_tasks)])

        # ---- 4. 编译模型 ----
        # 设置优化器（Adam lr=5e-4）、损失函数（8个BCE）、初始化权重、移动到GPU
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        # ---- Step 1: 特征输入 ----
        # 从batch中提取特征，返回 (batch_size, num_fields) 的索引矩阵
        X = self.get_inputs(inputs)

        # ---- Step 2: Embedding ----
        # 将特征索引映射为稠密向量，输出 (batch_size, num_fields, embedding_dim)
        feature_emb = self.embedding_layer(X)

        # ---- Step 3: 共享底层 ----
        # flatten: (batch_size, num_fields × embedding_dim)
        # bottom_output: (batch_size, bottom_hidden_units[-1])，如 (4096, 512)
        bottom_output = self.bottom(feature_emb.flatten(start_dim=1))

        # ---- Step 4: 8个任务Tower ----
        # 每个Tower独立处理共享底层输出，产生各自的logit
        tower_output = [self.tower[i](bottom_output) for i in range(self.num_tasks)]

        # ---- Step 5: 激活函数 ----
        # Sigmoid: logit → [0, 1] 概率，用于二分类
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        # ---- Step 6: 打包输出 ----
        # 返回字典 {"is_click_pred": tensor, "is_like_pred": tensor, ...}
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
