# ============================================================
# PLE_v12.py — PLE + Balance Loss + Uncertainty Weighting
# ============================================================
# 组合 PLE_v6 和 PLE_v7:
#   - Balance Loss (λ=0.01): 防止 Expert 极化
#   - Uncertainty Weighting: Kendall 可学习任务权重
# compute_loss = uncertainty_weighted_task_loss + balance_loss
# ============================================================

import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation


class CGC_Layer_v12(nn.Module):
    """与 CGC_Layer_v6/v7 相同，forward 返回 gate 权重"""
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim,
                 expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(CGC_Layer_v12, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks
        self.total_experts_per_task = num_specific_experts + num_shared_experts

        self.shared_experts = nn.ModuleList([
            MLP_Block(input_dim=input_dim, hidden_units=expert_hidden_units,
                      hidden_activations=hidden_activations, output_activation=None,
                      dropout_rates=net_dropout, batch_norm=batch_norm)
            for _ in range(self.num_shared_experts)
        ])
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                MLP_Block(input_dim=input_dim, hidden_units=expert_hidden_units,
                          hidden_activations=hidden_activations, output_activation=None,
                          dropout_rates=net_dropout, batch_norm=batch_norm)
                for _ in range(self.num_specific_experts)
            ]) for _ in range(num_tasks)
        ])
        self.gate = nn.ModuleList([
            MLP_Block(input_dim=input_dim,
                      output_dim=self.total_experts_per_task if i < num_tasks else num_shared_experts,
                      hidden_units=gate_hidden_units,
                      hidden_activations=hidden_activations,
                      output_activation=None,
                      dropout_rates=net_dropout,
                      batch_norm=batch_norm)
            for i in range(self.num_tasks + 1)
        ])
        self.gate_activation = get_activation('softmax')

    def forward(self, x):
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
        all_gates = []

        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                gate_input = torch.stack(
                    specific_expert_outputs[i] + shared_expert_outputs, dim=1
                )
                gate = self.gate_activation(self.gate[i](x[i]))
                all_gates.append(gate)
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            else:
                gate_input = torch.stack(shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[-1]))
                all_gates.append(gate)
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)

        return cgc_outputs, all_gates


class PLE_v12(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="PLE_v12",
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
                 balance_loss_weight=0.01,  # V6: Balance Loss λ
                 init_sigma=1.0,            # V7: log_sigma 初始值
                 **kwargs):
        super(PLE_v12, self).__init__(feature_map,
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
        self.balance_loss_weight = balance_loss_weight

        self.cgc_layers = nn.ModuleList([
            CGC_Layer_v12(num_shared_experts, num_specific_experts, num_tasks,
                          input_dim=embedding_dim * feature_map.num_fields if i == 0
                          else expert_hidden_units[-1],
                          expert_hidden_units=expert_hidden_units,
                          gate_hidden_units=gate_hidden_units,
                          hidden_activations=hidden_activations,
                          net_dropout=net_dropout,
                          batch_norm=batch_norm)
            for i in range(self.num_layers)
        ])
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

        # ===== V7: 可学习不确定性权重 =====
        init_log_sigma = torch.log(torch.tensor(init_sigma, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.full((num_tasks,), init_log_sigma))

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        cgc_inputs = [feature_emb.flatten(start_dim=1) for _ in range(self.num_tasks + 1)]

        all_layer_gates = []
        for i in range(self.num_layers):
            cgc_outputs, gates = self.cgc_layers[i](cgc_inputs)
            all_layer_gates.append(gates)
            cgc_inputs = cgc_outputs

        # ===== Balance Loss =====
        balance_loss = 0.0
        for layer_gates in all_layer_gates:
            for gate in layer_gates:
                u = gate.mean(dim=0)
                E = gate.shape[1]
                target = 1.0 / E
                balance_loss = balance_loss + torch.sum((u - target) ** 2)

        num_gates = sum(len(g) for g in all_layer_gates)
        balance_loss = self.balance_loss_weight * balance_loss / max(num_gates, 1)

        tower_output = [self.tower[i](cgc_outputs[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return_dict["balance_loss"] = balance_loss
        return return_dict

    def compute_loss(self, return_dict, y_true):
        """V12: Uncertainty Weighted Task Loss + Balance Loss"""
        labels = self.feature_map.labels
        # 计算各任务原始 loss
        raw_losses = [self.loss_fn[i](return_dict["{}_pred".format(labels[i])],
                                       y_true[i], reduction='mean')
                      for i in range(len(labels))]
        # Kendall 不确定性加权: L = Σ [1/(2σ²) * L_i + log(σ)]
        sigma_sq = torch.exp(2 * self.log_sigma)
        weighted_losses = [0.5 * raw_losses[i] / sigma_sq[i] + self.log_sigma[i]
                           for i in range(len(raw_losses))]
        loss = torch.sum(torch.stack(weighted_losses))
        loss += self.regularization_loss()
        # 叠加 Balance Loss
        if "balance_loss" in return_dict:
            loss = loss + return_dict["balance_loss"]
        return loss