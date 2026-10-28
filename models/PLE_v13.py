"""
PLE_v13: PLE + PPNet-style LHUC Gates
========================================
在 PLE 的 Task Tower 中注入 PPNet 的 LHUC (Learning Hidden Unit Contributions) 动态门控：
- 使用用户侧特征 (user_id, user_active_degree, is_live_streamer, is_video_author,
  is_lowactive_period, follow_user_num_range, fans_user_num_range, friend_user_num_range,
  register_days_range) 为每个 Task Tower 的每一层生成个性化门控
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
    每个隐藏层的激活值被用户特征生成的 gate 逐元素调制。
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
            # Gate: user_feat -> hidden_units[i+1], 用 2*sigmoid 确保门控以 1.0 为中心
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
        """
        x: (batch, input_dim)
        user_feat: (batch, user_feat_dim) - flattened user feature embeddings
        """
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.bn_layers[i](x)
            # LHUC gate: 2*sigmoid 中心在 1.0，既能放大也能抑制
            gate = 2 * torch.sigmoid(self.gate_nets[i](user_feat))
            x = x * gate
            if self.act_layers[i] is not None:
                x = self.act_layers[i](x)
            x = self.dropout_layers[i](x)
        x = self.output_layer(x)
        return x


class CGC_Layer(nn.Module):
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(CGC_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks
        self.shared_experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_shared_experts)])
        self.specific_experts = nn.ModuleList([nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_specific_experts)]) for _ in range(num_tasks)])
        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                             output_dim=num_specific_experts+num_shared_experts if i < num_tasks else num_shared_experts,
                                             hidden_units=gate_hidden_units,
                                             hidden_activations=hidden_activations,
                                             output_activation=None,
                                             dropout_rates=net_dropout,
                                             batch_norm=batch_norm) for i in range(self.num_tasks+1)])
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
        for i in range(self.num_tasks+1):
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


class PLE_v13(MultiTaskModel):
    """
    PLE + PPNet LHUC Gates
    - 在 PLE 的 CGC 层之后，每个 Task Tower 注入 PPNet 风格的 LHUC 门控
    - 门控由用户侧特征生成，实现个性化参数调制
    """
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="PLE_v13",
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
                 **kwargs):
        super(PLE_v13, self).__init__(feature_map,
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

        # CGC layers (same as original PLE)
        self.cgc_layers = nn.ModuleList([CGC_Layer(num_shared_experts,
                                                    num_specific_experts,
                                                    num_tasks,
                                                    input_dim=embedding_dim * feature_map.num_fields if i == 0 else expert_hidden_units[-1],
                                                    expert_hidden_units=expert_hidden_units,
                                                    gate_hidden_units=gate_hidden_units,
                                                    hidden_activations=hidden_activations,
                                                    net_dropout=net_dropout,
                                                    batch_norm=batch_norm) for i in range(self.num_layers)])

        # ===== PPNet: User feature indices in feature_emb =====
        # feature_map.json 中的顺序:
        # 0: user_id, 6: user_active_degree, 7: is_live_streamer, 8: is_video_author,
        # 12: is_lowactive_period, 13: follow_user_num_range, 14: fans_user_num_range,
        # 15: friend_user_num_range, 16: register_days_range
        self.user_feature_indices = [0, 6, 7, 8, 12, 13, 14, 15, 16]
        self.user_feat_dim = len(self.user_feature_indices) * embedding_dim  # 9 * emb_dim

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

        # CGC layers
        cgc_inputs = [feature_emb.flatten(start_dim=1) for _ in range(self.num_tasks + 1)]
        for i in range(self.num_layers):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs

        # ===== PPNet Tower: pass user_emb_flat for LHUC gating =====
        tower_output = [self.tower[i](cgc_outputs[i], user_emb_flat) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict