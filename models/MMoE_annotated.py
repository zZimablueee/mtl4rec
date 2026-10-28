# ============================================================
# MMoE.py — Multi-gate Mixture-of-Experts 多任务模型
# ============================================================
# 架构：Embedding → 8个专家MLP → 8个门控网络 → 8个加权组合 → 8个Tower → 8个预测
#
# 核心思想：不同于 SharedBottom 的「完全共享」，MMoE 用「选择性共享」：
#   - 8个专家 (Expert) 各自学习不同的特征表示
#   - 每个任务通过自己的门控 (Gate) 学习如何组合这些专家
#   - 任务间冲突时，不同任务可以选择不同的专家组合
#
# 与 SharedBottom 的关键区别：
#   SharedBottom: 所有任务 → 1个底层 → 各自Tower
#   MMoE: 所有任务 → 8个专家 → 各自门控加权组合 → 各自Tower
#
# 门控机制详解：
#   对每个任务 i：
#     1. Gate_i(x) → softmax → [w₁, w₂, ..., w₈]  (∑w = 1)
#     2. task_i_output = w₁·Expert₁(x) + w₂·Expert₂(x) + ... + w₈·Expert₈(x)
#   门控是一个小MLP，输入 x 输出 8 维权重，softmax 保证权重和为 1
#
# 配置参数（model_config_v3.yaml）：
#   num_experts: 8                        — 8个专家
#   expert_hidden_units: [512, 256, 128]  — 每个专家3层MLP
#   gate_hidden_units: [128, 64]          — 门控网络2层，输出8维
#   tower_hidden_units: [128, 64]         — 每个任务Tower 2层
# ============================================================

import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation
import logging


# ============================================================
# MMoE_Layer — 门控 + 专家组合层
# 这是 MMoE 的核心，替代了 SharedBottom 的单个共享底层
# ============================================================
class MMoE_Layer(nn.Module):
    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units,
                 gate_hidden_units, hidden_activations, net_dropout, batch_norm):
        super(MMoE_Layer, self).__init__()
        self.num_experts = num_experts  # 专家数量，本项目中 = 8
        self.num_tasks = num_tasks      # 任务数量，本项目中 = 8

        # ---- 专家网络 (Experts) ----
        # 每个专家是一个独立的 MLP，输入相同（原始特征），输出不同（各自学到的表示）
        # 为什么需要多个专家？不同任务需要不同特征组合：
        #   点击任务可能需要关注"视频时长+发布时间"，
        #   关注任务可能需要关注"作者粉丝数+历史关注率"，
        #   多个专家让不同任务可以"各取所需"
        self.experts = nn.ModuleList([
            MLP_Block(input_dim=input_dim,
                     hidden_units=expert_hidden_units,  # [512, 256, 128]
                     hidden_activations=hidden_activations,
                     output_activation=None,
                     dropout_rates=net_dropout,
                     batch_norm=batch_norm)
            for _ in range(self.num_experts)
        ])

        # ---- 门控网络 (Gates) ----
        # 每个任务有一个独立的门控网络
        # 门控是一个小MLP，输入原始特征，输出 num_experts 维权重
        # 经过 softmax 后，权重和为 1，表示每个专家对当前任务的贡献比例
        self.gate = nn.ModuleList([
            MLP_Block(input_dim=input_dim,
                     hidden_units=gate_hidden_units,  # [128, 64]
                     output_dim=num_experts,           # 输出 8 维，对应 8 个专家
                     hidden_activations=hidden_activations,
                     output_activation=None,  # 不做激活，后续用 softmax
                     dropout_rates=net_dropout,
                     batch_norm=batch_norm)
            for _ in range(self.num_tasks)
        ])

        # softmax 将门控输出转为概率分布（和为 1）
        self.gate_activation = get_activation('softmax')

    def forward(self, x):
        # ---- Step 1: 所有专家计算 ----
        # 8个专家分别对输入 x 计算，堆叠为 (batch, num_experts=8, expert_dim=128)
        experts_output = torch.stack(
            [self.experts[i](x) for i in range(self.num_experts)], dim=1
        )

        # ---- Step 2: 每个任务的门控加权组合 ----
        mmoe_output = []
        gate_output_list = []
        for i in range(self.num_tasks):
            # 2a. 门控网络计算权重
            gate_output = self.gate[i](x)           # (batch, 8)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # softmax，权重和为1

            gate_output_list.append(gate_output.mean(dim=0))  # 记录平均门控权重（用于分析）

            # 2b. 加权组合专家输出
            # gate_output: (batch, 8) → unsqueeze → (batch, 8, 1)
            # experts_output: (batch, 8, 128)
            # multiply → (batch, 8, 128) → sum(dim=1) → (batch, 128)
            # 含义：task_i 的输出 = Σ w_j × Expert_j(x)
            mmoe_output.append(
                torch.sum(
                    torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1
                )
            )

        return mmoe_output


# ============================================================
# MMoE 主模型
# ============================================================
class MMoE(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="MMoE",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_experts=4,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(MMoE, self).__init__(feature_map,
                                   task=task,
                                   num_tasks=num_tasks,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)

        # ---- 1. Embedding 层 ----
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_experts = num_experts

        # ---- 2. MMoE 层（核心） ----
        # 替代 SharedBottom 的 self.bottom
        # 输入维度 = embedding_dim × num_fields
        # 输出：8个任务的表示，每个 (batch, expert_hidden_units[-1]=128)
        self.mmoe_layer = MMoE_Layer(
            num_experts=num_experts,
            num_tasks=self.num_tasks,
            input_dim=embedding_dim * feature_map.num_fields,
            expert_hidden_units=expert_hidden_units,
            gate_hidden_units=gate_hidden_units,
            hidden_activations=hidden_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm
        )

        # ---- 3. 任务专属 Tower ----
        # 每个任务从 MMoE 层拿到自己的表示后，通过 Tower 产生最终预测
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=expert_hidden_units[-1],
                     output_dim=1,
                     hidden_units=tower_hidden_units,
                     hidden_activations=hidden_activations,
                     output_activation=None,
                     dropout_rates=net_dropout,
                     batch_norm=batch_norm)
            for _ in range(num_tasks)
        ])

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        # ---- Step 1: Embedding ----
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        # ---- Step 2: MMoE 层 ----
        # 输入: (batch, num_fields × embedding_dim) 的扁平向量
        # 输出: 8个任务各自的表示，每个 (batch, 128)
        # 关键：每个任务的表示是 8 个专家的加权组合，权重由各自的门控学习
        expert_output = self.mmoe_layer(feature_emb.flatten(start_dim=1))

        # ---- Step 3: 8个 Tower ----
        tower_output = [self.tower[i](expert_output[i]) for i in range(self.num_tasks)]

        # ---- Step 4: Sigmoid 激活 ----
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        # ---- Step 5: 打包输出 ----
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
