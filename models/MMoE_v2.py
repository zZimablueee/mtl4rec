"""
MMoE_v2: MMoE + PPNet-style LHUC Gates
========================================
在 MMoE 的 Task Tower 中注入 PPNet 的 LHUC (Learning Hidden Unit Contributions) 动态门控：
- 使用用户侧特征为每个 Task Tower 的每一层生成个性化门控
- 门控公式: tower_hidden = tower_hidden * (2 * sigmoid(gate(user_features)))
- 2*sigmoid 确保门控以 1.0 为中心，既能放大也能抑制
"""
import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation
import logging


class PPNetTower(nn.Module):
    """
    Task Tower with PPNet LHUC gates modulated by user features.
    """
    def __init__(self, input_dim, user_feat_dim, hidden_units, hidden_activations,
                 output_dim, dropout_rates, batch_norm):
        super(PPNetTower, self).__init__()
        hidden_units = [input_dim] + hidden_units
        self.num_layers = len(hidden_units) - 1
        self.layers = nn.ModuleList()
        self.gate_nets = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * self.num_layers

        hidden_activations = get_activation(hidden_activations, hidden_units)

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1], bias=True))
            self.gate_nets.append(nn.Sequential(
                nn.Linear(user_feat_dim, hidden_units[i + 1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_units[i + 1] // 2, hidden_units[i + 1])
            ))
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_units[i + 1]))
            else:
                self.bn_layers.append(nn.Identity())
            self.act_layers.append(hidden_activations[i])
            self.dropout_layers.append(nn.Dropout(dropout_rates[i]) if dropout_rates[i] > 0 else nn.Identity())

        self.output_layer = nn.Linear(hidden_units[-1], output_dim, bias=True)

    def forward(self, x, user_feat):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.bn_layers[i](x)
            gate = 2 * torch.sigmoid(self.gate_nets[i](user_feat))
            x = x * gate
            if self.act_layers[i] is not None:
                x = self.act_layers[i](x)
            x = self.dropout_layers[i](x)
        x = self.output_layer(x)
        return x


class MMoE_Layer(nn.Module):
    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(MMoE_Layer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_experts)])
        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                             hidden_units=gate_hidden_units,
                                             output_dim=num_experts,
                                             hidden_activations=hidden_activations,
                                             output_activation=None,
                                             dropout_rates=net_dropout,
                                             batch_norm=batch_norm) for _ in range(self.num_tasks)])
        self.gate_activation = get_activation('softmax')

    def forward(self, x):
        experts_output = torch.stack([self.experts[i](x) for i in range(self.num_experts)],
                                     dim=1)  # (?, num_experts, dim)
        mmoe_output = []
        gate_output_list = []
        for i in range(self.num_tasks):
            gate_output = self.gate[i](x)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # (?, num_experts)
            gate_output_list.append(gate_output.mean(dim=0))
            mmoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1))
        return mmoe_output


class MMoE_v2(MultiTaskModel):
    """
    MMoE + PPNet LHUC Gates
    - 在 MMoE 层之后，每个 Task Tower 注入 PPNet 风格的 LHUC 门控
    - 门控由用户侧特征生成，实现个性化参数调制
    """
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="MMoE_v2",
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
        super(MMoE_v2, self).__init__(feature_map,
                                      task=task,
                                      num_tasks=num_tasks,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_experts = num_experts

        # MMoE layer (same as original)
        self.mmoe_layer = MMoE_Layer(num_experts=num_experts,
                                     num_tasks=self.num_tasks,
                                     input_dim=embedding_dim * feature_map.num_fields,
                                     expert_hidden_units=expert_hidden_units,
                                     gate_hidden_units=gate_hidden_units,
                                     hidden_activations=hidden_activations,
                                     net_dropout=net_dropout,
                                     batch_norm=batch_norm)

        # ===== PPNet: User feature indices in feature_emb =====
        self.user_feature_indices = [0, 6, 7, 8, 12, 13, 14, 15, 16]
        self.user_feat_dim = len(self.user_feature_indices) * embedding_dim

        # ===== PPNet Tower: replace standard MLP towers with PPNetTower =====
        self.tower = nn.ModuleList([PPNetTower(
            input_dim=expert_hidden_units[-1],
            user_feat_dim=self.user_feat_dim,
            hidden_units=tower_hidden_units,
            hidden_activations=hidden_activations,
            output_dim=1,
            dropout_rates=net_dropout,
            batch_norm=batch_norm
        ) for _ in range(num_tasks)])

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)  # (batch, num_fields, emb_dim)

        # ===== PPNet: Extract user feature embeddings =====
        user_emb = feature_emb[:, self.user_feature_indices, :]  # (batch, num_user_features, emb_dim)
        user_emb_flat = user_emb.flatten(start_dim=1)  # (batch, num_user_features * emb_dim)

        # MMoE layer
        expert_output = self.mmoe_layer(feature_emb.flatten(start_dim=1))

        # ===== PPNet Tower: pass user_emb_flat for LHUC gating =====
        tower_output = [self.tower[i](expert_output[i], user_emb_flat) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict