"""
PLE_v14: PLE + User Bias (温和版个性化)
===========================================
与 V13 (PPNet per-layer LHUC gates) 的关键区别：
- V13: 每层 Tower 隐藏层都乘 2*sigmoid(gate)，破坏 CGC 专家分工
- V14: 只在最终 logit 上加法式偏置，不碰中间层，保留 CGC 完整能力

原理：用户特征 → 小 MLP → 标量偏置 → 加到 Tower logit → sigmoid
    logit = tower_mlp(cgc_output) + bias_scale * tanh(user_bias_mlp(user_features))

用户特征: user_id, user_active_degree, is_live_streamer, is_video_author,
          is_lowactive_period, follow_user_num_range, fans_user_num_range,
          friend_user_num_range, register_days_range
"""
import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation
import logging


class CGC_Layer(nn.Module):
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim,
                 expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(CGC_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks
        self.shared_experts = nn.ModuleList([MLP_Block(
            input_dim=input_dim, hidden_units=expert_hidden_units,
            hidden_activations=hidden_activations, output_activation=None,
            dropout_rates=net_dropout, batch_norm=batch_norm
        ) for _ in range(self.num_shared_experts)])
        self.specific_experts = nn.ModuleList([nn.ModuleList([MLP_Block(
            input_dim=input_dim, hidden_units=expert_hidden_units,
            hidden_activations=hidden_activations, output_activation=None,
            dropout_rates=net_dropout, batch_norm=batch_norm
        ) for _ in range(self.num_specific_experts)]) for _ in range(num_tasks)])
        self.gate = nn.ModuleList([MLP_Block(
            input_dim=input_dim,
            output_dim=num_specific_experts + num_shared_experts if i < num_tasks else num_shared_experts,
            hidden_units=gate_hidden_units, hidden_activations=hidden_activations,
            output_activation=None, dropout_rates=net_dropout, batch_norm=batch_norm
        ) for i in range(self.num_tasks + 1)])
        self.gate_activation = get_activation('softmax')

    def forward(self, x, require_gate=False):
        specific_expert_outputs = []
        shared_expert_outputs = []
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](x[i]))
            specific_expert_outputs.append(task_expert_outputs)
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))
        cgc_outputs = []
        gates = []
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                gate_input = torch.stack(specific_expert_outputs[i] + shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[i]))
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            else:
                gate_input = torch.stack(shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[-1]))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
        if require_gate:
            return cgc_outputs, gates
        else:
            return cgc_outputs


class PLE_v14(MultiTaskModel):
    """
    PLE + User Bias (温和个性化)
    - CGC 层完全保留，与 V5 一致
    - Tower 使用标准 MLP_Block，不做 per-layer 门控
    - 只在输出 logit 上加用户偏置：logit = tower_output + bias_scale * tanh(user_bias(user_features))
    - 对比 V13: 不破坏 CGC 专家分工，仅做个性化微调
    """
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="PLE_v14",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_layers=1,
                 num_shared_experts=1,
                 num_specific_experts=1,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 bias_scale=0.1,
                 bias_hidden_units=[64],
                 **kwargs):
        super(PLE_v14, self).__init__(feature_map,
                                       task=task,
                                       num_tasks=num_tasks,
                                       model_id=model_id,
                                       gpu=gpu,
                                       embedding_regularizer=embedding_regularizer,
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_layers = num_layers
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.bias_scale = bias_scale

        # CGC layers — 与 V5 完全一致
        self.cgc_layers = nn.ModuleList([CGC_Layer(
            num_shared_experts, num_specific_experts, num_tasks,
            input_dim=embedding_dim * feature_map.num_fields if i == 0 else expert_hidden_units[-1],
            expert_hidden_units=expert_hidden_units,
            gate_hidden_units=gate_hidden_units,
            hidden_activations=hidden_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm
        ) for i in range(self.num_layers)])

        # Tower — 使用标准 MLP_Block，与 V5 完全一致
        self.tower = nn.ModuleList([MLP_Block(
            input_dim=expert_hidden_units[-1],
            output_dim=1,
            hidden_units=tower_hidden_units,
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm
        ) for _ in range(num_tasks)])

        # ===== User Bias: 用户特征 → 标量偏置 =====
        self.user_feature_indices = [0, 6, 7, 8, 12, 13, 14, 15, 16]
        self.user_feat_dim = len(self.user_feature_indices) * embedding_dim

        # 每个任务一个小 MLP，输出标量用户偏置
        bias_layers = []
        hidden_units = [self.user_feat_dim] + bias_hidden_units
        for _ in range(num_tasks):
            layers = []
            for j in range(len(hidden_units) - 1):
                layers.append(nn.Linear(hidden_units[j], hidden_units[j + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_units[-1], 1))  # scalar bias
            bias_layers.append(nn.Sequential(*layers))
        self.user_bias_net = nn.ModuleList(bias_layers)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)  # (batch, num_fields, emb_dim)

        # 提取用户特征 embedding
        user_emb = feature_emb[:, self.user_feature_indices, :]  # (batch, 9, emb_dim)
        user_emb_flat = user_emb.flatten(start_dim=1)  # (batch, 9*emb_dim)

        # CGC 层 — 与 V5 完全一致
        cgc_inputs = [feature_emb.flatten(start_dim=1) for _ in range(self.num_tasks + 1)]
        for i in range(self.num_layers):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs

        # Tower — 标准 MLP
        tower_output = [self.tower[i](cgc_outputs[i]) for i in range(self.num_tasks)]

        # ===== User Bias: 加法式用户偏置（温和版） =====
        # 只在 logit 上加一个小偏移，不碰中间层
        y_pred = []
        for i in range(self.num_tasks):
            user_bias = self.user_bias_net[i](user_emb_flat)  # (batch, 1)
            logit = tower_output[i] + self.bias_scale * torch.tanh(user_bias)
            y_pred.append(self.output_activation[i](logit))

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict