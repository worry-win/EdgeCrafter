---
title: EC DINOv2-S 三层与五层 Decoder 设计说明
tags:
  - EdgeCrafter
  - DINOv2
  - Decoder
  - Detection
status: running
updated: 2026-08-09
---

# EC DINOv2-S 三层与五层 Decoder 设计说明

> [!summary] 结论
> 当前 5 层实验相对原严格 9 类的 3 层 DINOv2-S 基线，模型与训练配置的唯一实验变量是 `ECTransformer.num_layers: 3 -> 5`。Backbone、projection、HybridEncoder neck、query 数、deformable 采样方式、FDR、CDN、损失、优化器、数据增强、batch size、训练周期和随机种子均保持一致。
>
> “增加两层”指增加两套完整 decoder stage。每个新增 stage 都包含 TransformerDecoderLayer、独立分类头、132 维 FDR 回归头和 LQE，不是只增加两个没有预测头的 attention block。

## 1. 实验对象

| 项目 | 原 3 层基线 | 当前 5 层实验 |
|---|---|---|
| Backbone | DINOv2-S no-register | 相同 |
| 运行时 patch size | 16 | 相同 |
| Projection | EC 三尺度独立投影 | 相同 |
| Neck | HybridEncoder | 相同 |
| Decoder | ECTransformer，3 层 | ECTransformer，5 层 |
| Deformable levels | P3/P4/P5，共 3 层特征 | 相同 |
| 每头采样点 | `[3, 6, 3]` | 相同 |
| Normal queries | 300 | 相同 |
| CDN | 开启，`num_denoising=100` | 相同 |
| FDR | 开启，4 条边各 33 bins | 相同 |
| 类别数 | 严格 9 类 | 相同 |
| 训练 | 4 GPU，总 batch 32，每卡 8 | 相同 |
| Epoch / patience | 150 / 30 | 相同 |
| Seed | 42 | 相同 |

## 2. 配置文件选择

### 2.1 原 3 层基线

原 3 层严格 9 类实验使用：

- 基础模型与训练配置：[`ecdet_l_dinov2s_patch16_dec3_liver.yml`](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver.yml)
- 严格 9 类覆盖：[`ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml`](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml)
- Slurm：[`ecdet_ec_full_ignore9.sbatch`](../slurm/ecdet_ec_full_ignore9.sbatch)

配置继承关系：

```text
dataset/det_liver.yml + ecdet.yml
                 ↓
ecdet_l_dinov2s_patch16_dec3_liver.yml
                 ↓
ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml
```

其中第一层 liver 配置定义 DINOv2-S、EC projection、HybridEncoder、3 层 decoder、优化器和 150-epoch 训练计划；第二层只把训练与验证标注切换到严格 9 类 JSON，并设置 `num_classes=9`。

### 2.2 当前 5 层实验

当前实验使用：

- 配置：[`ecdet_l_dinov2s_patch16_dec5_liver_ignore9.yml`](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec5_liver_ignore9.yml)
- Slurm：[`ecdet_dinov2s_dec5_liver_ignore9_4gpu_150e.sbatch`](../slurm/ecdet_dinov2s_dec5_liver_ignore9_4gpu_150e.sbatch)
- Slurm Job：`842`

5 层配置直接继承严格 9 类的 3 层配置：

```yaml
__include__: [
  'ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml',
]

ECTransformer:
  num_layers: 5
```

配置中显式重申的 `eval_idx=-1`、`num_levels=3`、`num_points=[3,6,3]` 和 `cross_attn_method=default` 与基线继承值一致，不构成额外改动。

## 3. 加载的预训练权重

3 层和 5 层实验加载的是同一个 DINOv2-S no-register backbone 权重：

```text
/cobot/Code/xiangshaochong/checkpoints/dinov2/dinov2_vits14_pretrain.pth
```

权重特征：

| 属性 | 内容 |
|---|---|
| 模型 | 官方 DINOv2-S |
| ViT blocks | 12 |
| Token dim | 384 |
| Attention heads | 6 |
| 原始 patch size | 14 |
| Register token | 无 |
| 原始 patch projection | `[384, 3, 14, 14]` |

`DinoV2Adapter` 在加载时完成以下适配：

1. 删除只用于预训练的 `mask_token`。
2. 将 patch projection 从 patch14 双三次插值到 patch16。
3. 将位置编码插值到 `640 / 16 = 40` 的 patch 网格。
4. 严格加载 DINOv2-S backbone 参数。

> [!important] 初始化边界
> 这不是完整检测器 checkpoint。只有 `DinoV2Adapter.backbone` 加载预训练参数；三个 384→256 projection、HybridEncoder neck、decoder、分类头、回归头、criterion 和 EMA 都按照 seed 42 随机初始化。

训练命令首次启动时没有使用 `--tuning` 或完整检测器 `--resume`。只有同一输出目录已经存在 `last.pth` 时，Slurm 脚本才会恢复该实验自身的训练状态。

## 4. Backbone 到 Decoder 的数据流

输入尺寸为 `640×640`：

```text
Image [B, 3, 640, 640]
        ↓
DINOv2-S，运行时 patch16
        ↓
第 11、12 个 block 特征取平均
[B, 384, 40, 40]
        ↓ 双线性缩放 + 三个独立 1×1 projection
P3 [B, 256, 80, 80]
P4 [B, 256, 40, 40]
P5 [B, 256, 20, 20]
        ↓
HybridEncoder
        ↓
P3/P4/P5，均为 256 通道
        ↓ flatten
8400 个 multi-scale memory tokens
        ↓ encoder score + top-k
300 个 normal queries
        ↓
ECTransformer Decoder
```

HybridEncoder 配置保持不变：

| 参数 | 值 |
|---|---:|
| 输入/输出通道 | `[256, 256, 256]` |
| Strides | `[8, 16, 32]` |
| Transformer encoder | 仅 P5 上 1 层 |
| Hidden dim | 256 |
| FFN dim | 1024 |
| Expansion | 0.75 |
| Fuse op | sum |

## 5. 原 3 层 Decoder 网络设计

### 5.1 单层结构

每个 `TransformerDecoderLayer` 都执行：

```text
Query + reference-position embedding
        ↓
8-head query self-attention
        ↓ residual + LayerNorm
P3/P4/P5 memory
        ↓
MSDeformableAttention
  P3: 每个 head 采样 3 点
  P4: 每个 head 采样 6 点
  P5: 每个 head 采样 3 点
        ↓
Gate 门控融合
        ↓
FFN: 256 → 1024 → 256，SiLU
        ↓ residual + LayerNorm
        ↓
独立 9 类分类头 + 独立 FDR 回归头 + LQE
```

因此，原 3 层 decoder 从一开始就使用 `MSDeformableAttention`，并不是普通的全局 cross-attention。

### 5.2 分类头

默认配置为 `share_score_head=false`，所以 3 个 decoder layer 各有一个独立分类头：

```text
Layer 1: Linear(256, 9)
Layer 2: Linear(256, 9)
Layer 3: Linear(256, 9)
```

训练时三层结果均参与监督；推理时 `eval_idx=-1`，使用最后一层输出。

### 5.3 FDR 回归头

原 3 层 decoder 的每一层也都有独立的 132 维 FDR 回归头：

```text
reg_max = 32
每条边 = 32 + 1 = 33 bins
左、上、右、下 = 4 条边
输出维度 = 4 × 33 = 132
```

每层数据流为：

```text
query feature [256]
        ↓ MLP
edge logits [4, 33]
        ↓ softmax + Integral
四条边的连续距离 [4]
        ↓ distance2bbox
当前层 refined bbox
```

需要区分以下两个连续框头：

- `enc_bbox_head`：在 decoder 前生成候选框并选择 top-300。
- `pre_bbox_head`：第一层后生成初始参考框。

这两个头输出 4 维；decoder 内逐层 refinement 使用的是 132 维 FDR head。

### 5.4 三层展开

```text
Top-300 queries
   ↓
Decoder Layer 1
   ├─ score_head_1
   ├─ FDR_head_1
   └─ LQE_1
   ↓ refined reference boxes
Decoder Layer 2
   ├─ score_head_2
   ├─ FDR_head_2
   └─ LQE_2
   ↓ refined reference boxes
Decoder Layer 3
   ├─ score_head_3
   ├─ FDR_head_3
   └─ LQE_3
   ↓
Final prediction
```

最后一层同时作为 DDF 的 teacher distribution，前面层作为 student；FGL、DDF、MAL、box L1 和 GIoU 的开关及权重在 3 层与 5 层实验中完全一致。

## 6. 5 层相对 3 层的准确变化

解析完整 YAML 后，模型与训练配置的有效差异只有：

```diff
- ECTransformer.num_layers: 3
+ ECTransformer.num_layers: 5
```

由这个配置项自动产生的结构变化为：

| 模块 | 3 层 | 5 层 | 增量 |
|---|---:|---:|---:|
| TransformerDecoderLayer | 3 | 5 | +2 |
| 独立 decoder 分类头 | 3 | 5 | +2 |
| 独立 132 维 FDR 头 | 3 | 5 | +2 |
| LQE | 3 | 5 | +2 |
| Decoder auxiliary outputs | 2 组 | 4 组 | +2 组 |
| 训练态可训练参数 | 31,607,323 | 34,199,671 | +2,592,348 |
| Deploy 后参数（报告口径） | 31,364,873 | 33,949,777 | +2,584,904 |

这里存在两种正常但不能混用的统计口径：

- “训练态可训练参数”直接统计训练模型，包含各层辅助预测所需结构。
- 项目日志及消融总表中的 `Model Params` 先执行 `model.deploy()`，裁剪训练专用的辅助结构后再统计，因此数值较小。

旧报告中的 `31,364,873` 与当前 5 层任务日志中的 `33,949,777` 属于第二种口径；两者应直接比较。

没有变化的关键内容：

- DINOv2-S 权重及加载方式；
- DINOv2 blocks 11/12 特征选择与平均；
- P3/P4/P5 projection 和 HybridEncoder；
- hidden dim 256、decoder FFN 1024、8 heads；
- 每层 `MSDeformableAttention` 和 `[3,6,3]` 采样点；
- 300 normal queries、CDN、FDR、LQE、FGL、DDF 和 MAL；
- 9 类数据、输入 640、四卡总 batch 32；
- AdamW、学习率、weight decay、AMP、EMA、SyncBN；
- 150 epochs、patience 30、Mosaic/MixUp 时间表和 seed 42。

> [!note] 非实验差异
> 5 层任务使用单独输出目录，Slurm 同时允许 `batch,debug`，并将主机内存申请从旧脚本的 128 GB 提高到 192 GB。这些只影响文件隔离和资源调度，不改变模型前向、损失或优化过程。

## 7. 最终判断

当前 5 层实验可以作为严格的 decoder-depth 对照实验。准确说法是：

> 在完全相同的 DINOv2-S no-register backbone、EC projection、HybridEncoder、训练配置和检测算法下，把包含 deformable cross-attention、独立分类/FDR/LQE 头的 decoder stage 从 3 组增加到 5 组。

因此，可以将最终指标差异归因于 decoder 深度及其随层数增加的预测/refinement 能力和参数计算开销，而不是 backbone、neck、预训练权重或训练设置的变化。
