# mtl4rec

基于 KuaiRand-1K 数据集的多任务推荐（Multi-Task Learning）对比实验项目。

## 项目背景

在短视频推荐场景下，用户的反馈是多样的（点击、点赞、关注、评论、转发、完播等）。单任务模型只能优化单一目标，本项目在 KuaiRand-1K 上同时建模 **8 个任务**（is_click / is_like / is_follow / is_comment / is_forward / long_view / is_hate / is_profile_enter），探索多任务学习的架构演进与任务间权衡问题。

## 主要工作

- **架构对比**：实现并对比了 SharedBottom、MMoE、PLE 三代多任务架构，分析了 MMoE 的专家塌缩（expert collapse）问题，以及 PLE 通过专属/共享专家分离对该问题的缓解。
- **损失加权策略**：对比了静态任务权重、负载均衡损失（Balance Loss）、Kendall 不确定性加权三种方案，最终不确定性加权在小数据集上取得最优的平均 AUC。
- **排序评估**：除 AUC/gAUC 外，将多任务预测分数在 log 空间融合为统一排序分，按层级偏好构造 relevance，用 NDCG 评估融合排序质量。

## 目录结构

```
kuairand_preprocess.py   数据预处理（按用户划分 train/valid/test，生成 8 任务标签）
run_expid.py             训练入口
run_with_log.py          带日志记录的训练包装
end_to_end_eval.py       端到端评估：预测 → 多目标融合 → NDCG@K
improved_fusion_gpu.py   多任务分数融合（最终版）
config/                  数据集与模型配置
models/                  SharedBottom / MMoE / PLE 及各变体
fuxictr/                 基于 FuxiCTR 修改的训练框架
mtl-performance-table/   实验结果汇总页面
one_off_scripts/         一次性调试与分析脚本（归档）
data/ checkpoints/ results/   本地目录（数据与产物未上传，仅占位）
```

## 说明

- 数据集、模型权重与训练产物体积较大，未包含在本仓库中；数据集请从 [KuaiRand](https://kuairand.com/) 官方获取。
- 特征侧排除了交互级泄漏特征（播放时长、停留时长类字段），防止标签泄漏。

## 相关文档

- [mtl4rec（飞书项目文档）](https://mqw5c8smx9y.feishu.cn/wiki/Mm65wjr0Kid2qLkrQBPc9Oj8n7f)
- [MTL 文献梳理（飞书）](https://mqw5c8smx9y.feishu.cn/wiki/VVJKwm9GQiXDMSktIibctfXonVc)
