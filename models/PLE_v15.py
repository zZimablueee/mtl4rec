"""
PLE_v15: PLE + CGC Shared-Expert Residual Connection
======================================================
动机：V5 的 2 层 CGC 中，forward 任务专家被边缘化（AUC 从 0.713 崩到 0.592），
     但 comment 等稀疏任务受益。问题在于第 2 层 CGC 的 Gate 可以完全"关掉"
     某个任务的 shared expert 通道。

方案：在最后一层 CGC 中，强制每个任务的输出混入 shared expert 的均值：
    tower_input_k = Gate_k(experts) + alpha_k * mean(shared_experts)

- alpha_k 是可学习标量，per-task，初始化为 0.01
- 最差情况 alpha_k → 0，退化回 V5
- 不碰 Tower 层，不碰中间 CGC 层，只改最后一层
- 与 V13 (PPNet per-layer gate) 和 V14 (User Bias on logit) 的本质区别：
  在 expert 层动信息流，而非在 Tower 层或 logit 层

面试逻辑闭环："我发现 forward 边缘化 → 分析是 CGC gate 可完全关闭 shared 
  expert → 设计残差连接强制保留 shared 信号 → 验证效果"
"""
import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation


class CGC_Layer(nn.Module):
    """Standard CGC Layer (unchanged from PLE) — used for layers 0..N-2"""
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


class CGC_Layer_Residual(nn.Module):
    """
    CGC Layer with Shared-Expert Residual Connection.
    Only used for the LAST CGC layer.

    在标准 CGC gate 输出之上，强制加一条 shared expert 均值残差：
        output_k = Gate_k(experts) + alpha_k * mean(shared_experts)

    alpha_k: per-task learnable scalar, init 0.01
    如果某任务不需要 shared 信号，alpha_k 可以学到接近 0
    """
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim,
                 expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm, residual_alpha_init=0.01):
        super(CGC_Layer_Residual, self).__init__()
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

        # ===== Per-task learnable residual alpha =====
        self.residual_alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(residual_alpha_init)) for _ in range(num_tasks)
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

        # ===== Compute shared expert mean for residual =====
        shared_expert_mean = torch.stack(shared_expert_outputs, dim=1).mean(dim=1)  # (batch, dim)

        cgc_outputs = []
        gates = []
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                gate_input = torch.stack(specific_expert_outputs[i] + shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[i]))
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)

                # ===== RESIDUAL: add shared expert mean with per-task learnable alpha =====
                cgc_output = cgc_output + self.residual_alpha[i] * shared_expert_mean
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


class PLE_v15(MultiTaskModel):
    """
    PLE + CGC Shared-Expert Residual

    - 前 N-1 层 CGC: 标准 CGC_Layer，与 V5 完全一致
    - 最后一层 CGC: CGC_Layer_Residual，强制混入 shared expert 均值
    - Tower: 标准 MLP_Block，与 V5 完全一致
    - 不增参数量的主要部分（仅 N_task 个 alpha 标量）
    """
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="PLE_v15",
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
                 **kwargs):
        super(PLE_v15, self).__init__(feature_map,
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
            # 只有 1 层 → 直接用残差层
            self.cgc_layers = nn.ModuleList([])
            self.cgc_last = CGC_Layer_Residual(
                input_dim=embedding_dim * feature_map.num_fields,
                residual_alpha_init=residual_alpha_init,
                **cgc_kwargs
            )
        else:
            # 前 num_layers-1 层用标准 CGC
            self.cgc_layers = nn.ModuleList([CGC_Layer(
                input_dim=embedding_dim * feature_map.num_fields if i == 0 else expert_hidden_units[-1],
                **cgc_kwargs
            ) for i in range(num_layers - 1)])
            # 最后一层用残差 CGC
            self.cgc_last = CGC_Layer_Residual(
                input_dim=expert_hidden_units[-1],
                residual_alpha_init=residual_alpha_init,
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

        # 前 N-1 层标准 CGC
        for i in range(self.num_layers - 1):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs

        # 最后一层残差 CGC
        cgc_outputs = self.cgc_last(cgc_inputs)

        tower_output = [self.tower[i](cgc_outputs[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]

        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict