"""
PLE_v16: PLE + Task-Aware Gated Shared-Expert Residual Connection
==================================================================
动机：V15 的 Shared-Expert Residual 成功将 forward AUC 从 0.592 拉到 0.704 (+0.112)，
     但 comment 从 0.778 跌到 0.678 (-0.100)。问题在于 V15 对所有 task 无差别地
     注入残差，破坏了 comment 已建立的良好 expert-task 对齐。

方案：在 V15 的残差连接上增加 per-task 可学习 sigmoid gate β_k：
    tower_input_k = Gate_k(experts) + σ(β_k) * α_k * mean(shared_experts)

- σ(β_k): sigmoid gate，控制残差注入强度，per-task 可学习
- α_k: 残差缩放系数，与 V15 一致
- β_k 初始化策略（任务感知）：
  * is_forward: β=1.5 (σ≈0.82, 强残差) — 最被边缘化
  * is_hate: β=1.0 (σ≈0.73, 强残差) — 极端稀疏，V15 证明有效
  * is_click/follow/like: β=0.0 (σ=0.5, 中等) — 中性
  * is_comment: β=-1.0 (σ≈0.27, 弱残差) — 保护已对齐的稀疏任务
  * long_view/profile_enter: β=0.0 (σ=0.5, 中等)

与 V15 的区别：仅新增 N_task 个 β_k 参数 + sigmoid，参数量可忽略。
面试逻辑："V15 证明残差有效但无差别注入 → V16 加任务感知门控，选择性注入 →
          forward 收益保留 + comment 损失可控"
"""
import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation


class CGC_Layer(nn.Module):
    """Standard CGC Layer — used for layers 0..N-2, identical to V5/V15"""
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
                gate = self.gate_activation(self.gate[self.num_tasks](x[-1]))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
        if require_gate:
            return cgc_outputs, gates
        else:
            return cgc_outputs


class CGC_Layer_GatedResidual(nn.Module):
    """
    CGC Layer with Task-Aware Gated Shared-Expert Residual.
    Only used for the LAST CGC layer.

    V15: output_k = Gate_k(experts) + alpha_k * mean(shared_experts)
    V16: output_k = Gate_k(experts) + sigmoid(beta_k) * alpha_k * mean(shared_experts)

    - alpha_k: residual scale (from V15), per-task learnable, init 0.01
    - beta_k:  residual gate (V16 NEW), per-task learnable, task-aware init
      * logit space, fed through sigmoid to get [0, 1] gate value
      * init: forward=1.5, hate=1.0, comment=-1.0, others=0.0
    """
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim,
                 expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm, residual_alpha_init=0.01,
                 residual_gate_init=None):
        super(CGC_Layer_GatedResidual, self).__init__()
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

        # ===== V15: residual alpha =====
        self.residual_alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(residual_alpha_init)) for _ in range(num_tasks)
        ])

        # ===== V16 NEW: residual gate beta (logit space, fed through sigmoid) =====
        if residual_gate_init is None:
            # Default: all 0.0 → sigmoid(0) = 0.5
            residual_gate_init = [0.0] * num_tasks
        self.residual_gate = nn.ParameterList([
            nn.Parameter(torch.tensor(residual_gate_init[i])) for i in range(num_tasks)
        ])

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

        # ===== Shared expert mean for residual =====
        shared_expert_mean = torch.stack(shared_expert_outputs, dim=1).mean(dim=1)

        cgc_outputs = []
        gates = []
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                gate_input = torch.stack(specific_expert_outputs[i] + shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[i]))
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)

                # ===== V16: GATED RESIDUAL =====
                gate_value = torch.sigmoid(self.residual_gate[i])  # [0, 1]
                cgc_output = cgc_output + gate_value * self.residual_alpha[i] * shared_expert_mean
                cgc_outputs.append(cgc_output)
            else:
                gate_input = torch.stack(shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[self.num_tasks](x[-1]))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)

        if require_gate:
            return cgc_outputs, gates
        else:
            return cgc_outputs


class PLE_v16(MultiTaskModel):
    """
    PLE + Task-Aware Gated Shared-Expert Residual Connection

    V15 → V16 改进:
    - V15: 对所有 task 无差别注入残差 → forward +0.112, comment -0.100
    - V16: 加 sigmoid gate β_k 选择性注入 → 期望 forward 收益保留 + comment 损失可控

    结构:
    - 前 N-1 层 CGC: 标准 CGC_Layer，与 V5/V15 完全一致
    - 最后一层 CGC: CGC_Layer_GatedResidual，V16 新设计
    - Tower: 标准 MLP_Block，与 V5/V15 完全一致
    """
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="PLE_v16",
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
                 residual_alpha_init=0.01,
                 residual_gate_init=None,
                 **kwargs):
        super(PLE_v16, self).__init__(feature_map,
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

        cgc_kwargs = dict(
            num_shared_experts=num_shared_experts,
            num_specific_experts=num_specific_experts,
            num_tasks=num_tasks,
            expert_hidden_units=expert_hidden_units,
            gate_hidden_units=gate_hidden_units,
            hidden_activations=hidden_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm
        )

        if num_layers == 1:
            self.cgc_layers = nn.ModuleList([])
            self.cgc_last = CGC_Layer_GatedResidual(
                input_dim=embedding_dim * feature_map.num_fields,
                residual_alpha_init=residual_alpha_init,
                residual_gate_init=residual_gate_init,
                **cgc_kwargs
            )
        else:
            self.cgc_layers = nn.ModuleList([CGC_Layer(
                input_dim=embedding_dim * feature_map.num_fields if i == 0 else expert_hidden_units[-1],
                **cgc_kwargs
            ) for i in range(num_layers - 1)])
            self.cgc_last = CGC_Layer_GatedResidual(
                input_dim=expert_hidden_units[-1],
                residual_alpha_init=residual_alpha_init,
                residual_gate_init=residual_gate_init,
                **cgc_kwargs
            )

        self.tower = nn.ModuleList([MLP_Block(
            input_dim=expert_hidden_units[-1],
            output_dim=1,
            hidden_units=tower_hidden_units,
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm
        ) for _ in range(num_tasks)])

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        cgc_inputs = [feature_emb.flatten(start_dim=1) for _ in range(self.num_tasks + 1)]

        for i in range(self.num_layers - 1):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs

        cgc_outputs = self.cgc_last(cgc_inputs)

        tower_output = [self.tower[i](cgc_outputs[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict