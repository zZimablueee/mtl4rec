# ============================================================
# PLE.py — Progressive Layered Extraction 多任务模型
# ============================================================
# 架构：Embedding → CGC层1 → CGC层2 → 8个Tower → 8个预测
# 参考论文：Tang et al., "Progressive Layered Extraction"
#
# 核心思想：在 MMoE 基础上增加三个改进：
#   1. 任务专属专家 (Task-Specific Experts) — 每个任务有私有专家，硬隔离
#   2. 共享专家 (Shared Experts) — 学习共性模式，所有任务共享
#   3. 渐进式提取 (Progressive) — 多层 CGC 堆叠，逐层提取高阶特征
#
# 与 MMoE 的关键区别：
#   MMoE: 所有专家全共享 → 软隔离（门控降低权重）
#   PLE:  专属专家 + 共享专家 → 硬隔离 + 软选择
#
# CGC 层详解（以本项目配置为例）：
#   - 4 个共享专家 (Shared₁~Shared₄) — 所有任务可见
#   - 每任务 2 个专属专家 (S_i₁, S_i₂) — 仅任务 i 可见
#   - 每任务 1 个门控：看到 2+4=6 个专家，输出 6 维 softmax 权重
#   - 1 个共享门控：看到 4 个共享专家，输出 4 维 softmax 权重
#   - 门控输入是原始特征 x（矢量），输出是标量权重
#   - 这和之前 MMoE 的门控机制完全一致：矢量入 → 标量出
#
# 配置参数（model_config_v7.yaml）：
#   num_layers: 2                 — 2层CGC堆叠
#   num_shared_experts: 4         — 4个共享专家
#   num_specific_experts: 2       — 每任务2个专属专家
#   expert_hidden_units: [512,256,128] — 每个专家3层MLP
#   gate_hidden_units: [128, 64]  — 门控2层，输出维度=专属+共享
#   tower_hidden_units: [128, 64] — 每个任务Tower 2层
# ============================================================

import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation
import logging


# ============================================================
# CGC_Layer — Customized Gate Control 层
# 这是 PLE 的核心组件，替代了 MMoE 的 MMoE_Layer
# 一层 CGC 包含：专属专家 + 共享专家 + 门控网络
# ============================================================
class CGC_Layer(nn.Module):
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks,
                 input_dim, expert_hidden_units, gate_hidden_units,
                 hidden_activations, net_dropout, batch_norm):
        super(CGC_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts      # 4
        self.num_specific_experts = num_specific_experts  # 2
        self.num_tasks = num_tasks                        # 8

        # ---- 共享专家 (Shared Experts) ----
        # 所有任务共享，学习共性模式
        # 4个专家，每个 512→256→128
        self.shared_experts = nn.ModuleList([
            MLP_Block(input_dim=input_dim,
                     hidden_units=expert_hidden_units,
                     hidden_activations=hidden_activations,
                     output_activation=None,
                     dropout_rates=net_dropout,
                     batch_norm=batch_norm)
            for _ in range(self.num_shared_experts)
        ])

        # ---- 专属专家 (Task-Specific Experts) ----
        # 每个任务有 num_specific_experts=2 个专属专家
        # 其他任务的门控无法看到这些专家 → 硬隔离
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                MLP_Block(input_dim=input_dim,
                         hidden_units=expert_hidden_units,
                         hidden_activations=hidden_activations,
                         output_activation=None,
                         dropout_rates=net_dropout,
                         batch_norm=batch_norm)
                for _ in range(self.num_specific_experts)
            ]) for _ in range(num_tasks)
        ])

        # ---- 门控网络 (Gates) ----
        # 前 num_tasks 个门控：每个任务的门控，输出维度 = 专属 + 共享（2+4=6）
        # 最后一个门控：共享门控，输出维度 = 共享专家数（4）
        # 门控输入是原始特征 x（矢量），输出是标量权重（和 MMoE 一致）
        self.gate = nn.ModuleList([
            MLP_Block(input_dim=input_dim,
                     output_dim=num_specific_experts + num_shared_experts
                                if i < num_tasks else num_shared_experts,
                     hidden_units=gate_hidden_units,  # [128, 64]
                     hidden_activations=hidden_activations,
                     output_activation=None,  # 不做激活，后续 softmax
                     dropout_rates=net_dropout,
                     batch_norm=batch_norm)
            for i in range(self.num_tasks + 1)
        ])
        self.gate_activation = get_activation('softmax')

    def forward(self, x):
        """
        x: list of tensors, len = num_tasks + 1
            x[0]~x[7]: 每个任务的输入矢量（第一层是原始特征，后面是上一层输出）
            x[8]: 共享输入矢量
        """
        # ---- Step 1: 专属专家计算 ----
        # 每个任务用自己的专属专家处理自己的输入
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](x[i]))
            specific_expert_outputs.append(task_expert_outputs)
            # task_expert_outputs: [S_i1, S_i2]，每个是 (batch, 128)

        # ---- Step 2: 共享专家计算 ----
        shared_expert_outputs = []
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))
        # shared_expert_outputs: [Shared₁, Shared₂, Shared₃, Shared₄]

        # ---- Step 3: 门控加权组合 ----
        cgc_outputs = []
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                # 任务门控：看到专属专家[2] + 共享专家[4] = 6个专家
                gate_input = torch.stack(
                    specific_expert_outputs[i] + shared_expert_outputs, dim=1
                )  # (batch, 6, 128)
                # 门控输入是 x[i]（矢量），输出是 6 维权重（标量矢量）
                gate = self.gate_activation(self.gate[i](x[i]))  # (batch, 6)
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)  # (batch, 128)
                cgc_outputs.append(cgc_output)
            else:
                # 共享门控：只看到共享专家[4]
                gate_input = torch.stack(shared_expert_outputs, dim=1)  # (batch, 4, 128)
                gate = self.gate_activation(self.gate[i](x[-1]))  # (batch, 4)
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)  # (batch, 128)
                cgc_outputs.append(cgc_output)

        return cgc_outputs
        # 返回: [task_0_rep, ..., task_7_rep, shared_rep]
        # 每个是 (batch, 128)，作为下一层 CGC 的输入


# ============================================================
# PLE 主模型
# ============================================================
class PLE(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="PLE",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_layers=1,           # CGC 层数，本项目中 = 2
                 num_shared_experts=1,   # 共享专家数，本项目中 = 4
                 num_specific_experts=1, # 每任务专属专家数，本项目中 = 2
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(PLE, self).__init__(feature_map,
                                   task=task,
                                   num_tasks=num_tasks,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)

        # ---- 1. Embedding 层 ----
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_layers = num_layers
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts

        # ---- 2. CGC 层堆叠 (渐进式提取) ----
        # 多层 CGC 逐层提取高阶特征
        # 第一层 input_dim = embedding_dim × num_fields（如 800 维）
        # 后续层 input_dim = expert_hidden_units[-1]（128 维）
        self.cgc_layers = nn.ModuleList([
            CGC_Layer(num_shared_experts,
                     num_specific_experts,
                     num_tasks,
                     input_dim=embedding_dim * feature_map.num_fields
                              if i == 0 else expert_hidden_units[-1],
                     expert_hidden_units=expert_hidden_units,
                     gate_hidden_units=gate_hidden_units,
                     hidden_activations=hidden_activations,
                     net_dropout=net_dropout,
                     batch_norm=batch_norm)
            for i in range(self.num_layers)
        ])

        # ---- 3. 任务专属 Tower ----
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

        # ---- 4. 编译模型 ----
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        # ---- Step 1: Embedding ----
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        # ---- Step 2: 初始化 CGC 输入 ----
        # 第一层: 所有任务和共享都使用相同的原始特征
        cgc_inputs = [feature_emb.flatten(start_dim=1) for _ in range(self.num_tasks + 1)]

        # ---- Step 3: 多层 CGC 渐进式提取 ----
        for i in range(self.num_layers):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs  # 这一层输出 → 下一层输入

        # ---- Step 4: 8个 Tower ----
        tower_output = [self.tower[i](cgc_outputs[i]) for i in range(self.num_tasks)]

        # ---- Step 5: Sigmoid 激活 ----
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        # ---- Step 6: 打包输出 ----
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
