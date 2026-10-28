# ============================================================
# PLE_stability.py — PLE + 训练稳定性指标记录
# ============================================================
# 基于 PLE (V5: 2层CGC + 静态loss_weight)
# 新增: 每step记录梯度范数(裁剪前) + per-task loss
# ============================================================

import torch
from torch import nn
import json
import os
import sys
import logging
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation
from tqdm import tqdm


class CGC_Layer_stability(nn.Module):
    """与 PLE.py 中 CGC_Layer 完全一致"""
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim,
                 expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(CGC_Layer_stability, self).__init__()
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
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                gate_input = torch.stack(specific_expert_outputs[i] + shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[i]))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            else:
                gate_input = torch.stack(shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[-1]))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
        return cgc_outputs


class PLE_stability(MultiTaskModel):
    def __init__(self,
                 feature_map,
                 task=["binary_classification"],
                 num_tasks=1,
                 model_id="PLE_stability",
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
                 metrics_path=None,
                 **kwargs):
        super(PLE_stability, self).__init__(
            feature_map=feature_map,
            task=task,
            num_tasks=num_tasks,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs
        )
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_layers = num_layers
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.cgc_layers = nn.ModuleList([
            CGC_Layer_stability(
                num_shared_experts, num_specific_experts, num_tasks,
                input_dim=embedding_dim * feature_map.num_fields if i == 0 else expert_hidden_units[-1],
                expert_hidden_units=expert_hidden_units,
                gate_hidden_units=gate_hidden_units,
                hidden_activations=hidden_activations,
                net_dropout=net_dropout,
                batch_norm=batch_norm
            ) for i in range(self.num_layers)
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
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

        # 稳定性指标记录
        self._metrics_path = metrics_path
        self._stability_metrics = {'model': 'PLE_static_loss', 'epochs': []}
        self._step_metrics = []
        self._step_count = 0
        self._epoch_count = 0

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        cgc_inputs = [feature_emb.flatten(start_dim=1) for _ in range(self.num_tasks + 1)]
        for i in range(self.num_layers):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs
        tower_output = [self.tower[i](cgc_outputs[i]) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

    def train_step(self, batch_data):
        """覆盖 train_step: 记录梯度范数(裁剪前) + per-task losses"""
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()

        # 记录梯度范数 (裁剪前)
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()

        # 记录 per-task losses
        labels = self.feature_map.labels
        with torch.no_grad():
            task_losses = [
                self.loss_fn[i](
                    return_dict["{}_pred".format(labels[i])],
                    y_true[i], reduction='mean'
                ).item()
                for i in range(len(labels))
            ]

        self._step_metrics.append({
            'step': self._step_count,
            'loss': loss.item(),
            'per_task_loss': task_losses,
            'grad_norm': total_norm,
        })
        self._step_count += 1
        return loss

    def train_epoch(self, data_generator):
        """覆盖 train_epoch: 每epoch保存指标 + 保留原有逻辑"""
        self._step_metrics = []
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

        # 保存本 epoch 指标
        epoch_data = {
            'epoch': self._epoch_count,
            'steps': list(self._step_metrics),
        }
        self._stability_metrics['epochs'].append(epoch_data)
        self._epoch_count += 1

        if self._metrics_path:
            os.makedirs(os.path.dirname(self._metrics_path), exist_ok=True)
            with open(self._metrics_path, 'w') as f:
                json.dump(self._stability_metrics, f)