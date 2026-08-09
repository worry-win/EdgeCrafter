---
title: EdgeCrafter 检测消融实验报告
date: 2026-08-07
status: in-progress
dataset: det_liver
task: detection
tags:
  - EdgeCrafter
  - ablation
  - detection
  - det-liver
---

# EdgeCrafter 检测消融实验报告

关联文档：[[EC_ABLATIONS]] · [[EC_CODEX_HANDOFF]] · [[LESION_DETECTION_DATASET_STATS]]

> [!important] 结果口径
> 结果表统一使用各已完成实验的最终评估结果。`mAP50` 表示 AP@[IoU=0.50]，`mAP50-95` 表示 COCO AP@[IoU=0.50:0.95]；F1 是 IoU50 PR 曲线上的最大 macro-F1。P/R 是同一最大 F1 工作点的 macro precision/recall。尚未完成的实验保留空白结果单元格。

## 统一实验口径

所有实验以严格 9 类 `EC-full` 为对照，除各节列出的消融项外，其余配置保持一致。

| 项目 | 统一配置 |
|---|---|
| 数据集 | `det_liver` |
| 类别 | `num_classes=9`，类别 ID `0..8` |
| train 标注 | `/cobot/Data/Lesion_det/det_liver/annotations/train_ignore_9_12.json` |
| valid 标注 | `/cobot/Data/Lesion_det/det_liver/annotations/valid_ignore_9_12.json` |
| 输入尺寸 | `640 × 640` |
| Backbone | 无 register 的 DINOv2-S，12 blocks，384 dim，6 heads，运行时 patch size 16 |
| EC-full 特征路径 | EC 三尺度 projection → HybridEncoder → ECTransformer |
| Decoder | 3 层，300 normal queries |
| 优化器 | AdamW；backbone LR `5e-6`，其余主 LR `5e-4` |
| 训练计划 | 150 epochs，warmup 2,000 iterations，patience 30 |
| Batch size | 总 batch size 32，4 GPU |
| 训练设置 | seed 42、AMP、SyncBN、EMA |
| EC-full 分类损失 | MAL，`alpha=0.75`，`gamma=1.5` |
| EC-full定位路径 | FDR + L1 + GIoU + FGL + DDF + GO union |
| EC-full CDN | `num_denoising=100` |
| EC-full增强 | Mosaic 与 MixUp 在前 24 epochs 开启，最后 2 epochs 关闭强增强 |

基线配置：[ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml)

### 当前任务状态（2026-08-07）

| 实验 | Slurm job | 状态 |
|---|---:|---|
| EC-full strict-9 | 直接 GPU 运行 | 已完成 |
| no-CDN strict-9 | 749 | 已完成 |
| no-FDR decode strict-9 | 752 / 766 | `best.pth` 评估已完成；训练任务仍在运行 |
| no-GO-DDF strict-9 | 750 / 765 | `best.pth` 评估已完成 |
| no-FDR + no-GO-DDF strict-9 | 751 / 764 | `best.pth` 评估已完成；训练任务仍在运行 |
| no-Mosaic + no-MAL strict-9 | 756 / 768 | `best.pth` 评估已完成；训练任务仍在运行 |
| RF P4 neck strict-9 | 直接 GPU 运行 | 已完成 |

## CDN 消融

### 1. 实验目的

评估 Contrastive Denoising（CDN）训练分支对收敛和检测精度的独立贡献。CDN 只在训练期添加带标签噪声和框噪声的 DN queries，不改变最终 300 个 normal queries 的推理输出。

### 2. 算法实现细节

EC-full 设置 `num_denoising=100`。训练时，decoder 根据 GT 创建正负 noisy queries，并构造 attention mask，使 normal queries 看不到 DN queries、不同 DN group 之间相互隔离。decoder 输出 `dn_outputs`、`dn_pre_outputs` 和 `dn_meta` 后，criterion 使用固定的 DN positive indices 计算 MAL、box、FGL/DDF 等 DN losses。

`no-CDN` 只设置：

```yaml
ECTransformer:
  num_denoising: 0
```
关闭后：

- 不创建 `denoising_class_embed`；
- 不调用 `get_contrastive_denoising_training_group()`；
- 不拼接 DN queries，也不构造 DN attention mask；
- model output 不含 `dn_outputs` 和 `dn_meta`；
- normal queries、MAL、GO union、FDR、FGL、DDF 和推理路径保持不变。

实现入口：[decoder.py](../ecdetseg/engine/edgecrafter/decoder.py) · [denoising.py](../ecdetseg/engine/edgecrafter/denoising.py) · [criterion.py](../ecdetseg/engine/edgecrafter/criterion.py)

### 3. 实验具体配置

| 实验 | 配置文件 | `num_denoising` | 参数量 | 输出目录 |
|---|---|---:|---:|---|
| EC-full | `..._ec_full_ignore9.yml` | 100 | 31,364,873 | `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9` |
| no-CDN | `..._no_cdn_ignore9.yml` | 0 | 31,362,313 | `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_ignore9` |

消融配置：[ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_ignore9.yml](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_ignore9.yml)

### 4. 实验结果与分析

| 实验 | Model Params | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|---:|
| EC-full | 31,364,873 | 0.594 | 0.593 | 0.281 | 0.598 | 0.590 |
| no-CDN | 31,362,313 | 0.594 | 0.589 | 0.270 | 0.570 | 0.620 |

当前单 seed 结果中，关闭 CDN 后 mAP50-95 下降 `0.011`，mAP50 下降 `0.004`；recall 从 `0.590` 提高到 `0.620`，precision 从 `0.598` 降到 `0.570`，F1 基本不变。CDN 仅减少 2,560 个参数，因此参数量几乎不变是符合实现预期的：它主要改变训练 queries 和损失，不是推理主干模块。

> [!caution] 结论边界
> 该差异较小，目前只能说明 CDN 在这一训练口径下没有表现出稳定的正向增益。至少需要补充多个 seed，才能区分真实模块效应和训练随机波动。

## D-FINE decoder 消融

### 1. 实验目的

拆分 D-FINE decoder 中三个相互关联但不等价的机制：

1. FDR 分布式边界解码；
2. GO union matching；
3. FGL/DDF 分布监督。

实验避免把“只改变最终框解码方式”和“完全移除分布回归路径”混称为同一个 D-FINE 消融。

### 2. 算法实现细节

#### FDR

EC-full 的每层 decoder 输出四条离散边界分布 `pred_corners`，通过 Integral/FDR 转成 box，并由 LQE 用分布质量修正分类分数。

`no-FDR decode` 设置 `use_fdr_decode=false`，使用独立的连续 4 维 box head 生成最终框；同时保留 `use_aux_distribution=true`，继续产生 `pred_corners`，因此 FGL 和 DDF 仍然有效。这个实验是最终解码路径对照，不是完整移除分布 head。

#### GO union 与 DDF

EC-full 对 final decoder、auxiliary decoder、pre-output 和 encoder output 分别进行 Hungarian matching，再由 `_get_go_indices()` 合并成定位损失使用的 union set。`use_uni_set=false` 后，各层定位损失使用自己的 layer-local matching。

DDF 使用当前层的边界分布学习 teacher distribution；`use_ddf=false` 后不再产生 `loss_ddf*`，但 `use_fgl=true` 时仍保留 GT 边界分布监督。

#### 连续回归组合

`no-FDR + no-GO-DDF` 同时关闭：

- FDR 最终框解码；
- distribution auxiliary output；
- LQE；
- FGL；
- DDF；
- GO union matching。

criterion 只保留 `['mal', 'boxes']`，decoder 使用连续 4 维 box regression。这一组才是当前矩阵中的 continuous-box-only 对照。

### 3. 实验具体配置

| 实验                 | FDR decode | 分布辅助 | LQE | GO union | FGL | DDF | 分类损失 | 输出目录                           |
| ------------------ | :--------: | :--: | :-: | :------: | :-: | :-: | ---- | ------------------------------ |
| EC-full            |     开      |  开   |  开  |    开     |  开  |  开  | MAL  | `..._ec_full_ignore9`          |
| no-FDR decode      |     关      |  开   |  关  |    开     |  开  |  开  | MAL  | `..._no_fdr_decode_ignore9`    |
| no-GO-DDF          |     开      |  开   |  开  |    关     |  开  |  关  | MAL  | `..._no_go_ddf_ignore9`        |
| no-FDR + no-GO-DDF |     关      |  关   |  关  |    关     |  关  |  关  | MAL  | `..._no_fdr_no_go_ddf_ignore9` |

配置文件：

- [no_fdr_decode_ignore9.yml](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_decode_ignore9.yml)
- [no_go_ddf_ignore9.yml](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_go_ddf_ignore9.yml)
- [no_fdr_no_go_ddf_ignore9.yml](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_no_go_ddf_ignore9.yml)

### 4. 实验结果与分析

| 实验 | Model Params | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|---:|
| EC-full | 31,364,873 | 0.594 | 0.593 | 0.281 | 0.598 | 0.590 |
| no-FDR decode | 31,761,300 | 0.598 | 0.601 | 0.283 | 0.569 | 0.630 |
| no-GO-DDF | 31,364,873 | 0.592 | 0.592 | 0.282 | 0.574 | 0.610 |
| no-FDR + no-GO-DDF | 31,264,776 | 0.588 | 0.588 | 0.278 | 0.559 | 0.620 |

#### AP50 训练曲线

![[assets/d_fine_ablations_map50_training_curve.png]]

三个结果均由各自 `best.pth` 在严格 9 类评估集上独立得到，评估日志保存在对应实验目录的 `test_strict9/eval.log`。

- `no-FDR decode` 的 mAP50 为 `0.601`、mAP50-95 为 `0.283`，相对 EC-full 分别为 `+0.008`、`+0.002`，但 precision 从 `0.598` 降至 `0.569`，recall 从 `0.590` 升至 `0.630`。该配置同时关闭 LQE 并新增连续回归 head，且仍保留 FGL/DDF，因此只能说明当前的最终连续解码路径表现不弱，不能单独判定 FDR 分布监督无效。
- `no-GO-DDF` 与 EC-full 参数量相同，mAP50 几乎持平（`-0.001`），mAP50-95 略升 `0.001`，F1 下降 `0.002`。在当前单 seed 训练下，GO union 与 DDF 的独立收益较小。
- 完整连续回归对照 `no-FDR + no-GO-DDF` 的 mAP50、mAP50-95 和 F1 分别下降 `0.005`、`0.003` 和 `0.006`。这表明分布回归及 GO-DDF 的组合存在小幅正向贡献，主要体现为更高的 precision；但该组合实验同时移除了多个机制，不能把差异归因到其中任一项。

> [!caution] 结论边界
> 三组差异均处于较小范围，且目前每种配置仅有一个 seed。应以至少三个相同训练口径的 seed 报告均值和标准差，再判断 FDR、GO 或 DDF 的稳定贡献。

## EC 减去 Mosaic 增强和 MAL Loss 消融

### 1. 实验目的

评估同时移除 Mosaic 数据增强与 MAL 分类目标后，对 EC-full 检测性能的组合影响。该实验回答的是“两项策略共同移除”的效果，不能单独归因到 Mosaic 或 MAL。

### 2. 算法实现细节

#### 关闭 Mosaic

设置 `mosaic_prob=0.0`。Mosaic transform 仍存在于 Compose 中，但调度条件恒为 false，因此不会进入 Mosaic forward。

关闭 Mosaic 不等于关闭全部强增强：

- MixUp 仍在前 24 epochs 保持 `mixup_prob=1.0`；
- RandomPhotometricDistort 和 RandomHorizontalFlip 保留；
- Mosaic 与 RandomZoomOut/RandomIoUCrop 在实现中互斥，因此关闭 Mosaic 后，ZoomOut 和 IoU Crop 会继续执行；
- 训练计划、优化器和最后 2 epochs 的 no-augmentation 口径不变。

#### 关闭 MAL

MAL 使用匹配框 IoU 的 `gamma` 次幂作为正样本分类软目标，并用 detach 后的预测分数构造负样本权重。关闭 MAL 时不能直接删除分类损失，否则模型将失去分类监督。

本实验将 MAL 替换为标准 sigmoid focal loss：

```yaml
ECCriterion:
  losses: ['focal', 'boxes', 'local']
  alpha: 0.25
  gamma: 2.0
```

`loss_focal` 权重为 1；L1、GIoU、FGL、DDF、GO union 和 CDN 保持开启。focal one-hot target 会转换为与 logits 相同的浮点 dtype，以兼容 AMP。

### 3. 实验具体配置

| 实验 | Mosaic | MixUp | 分类损失 | FGL/DDF | GO union | CDN | 输出目录 |
|---|:---:|:---:|---|:---:|:---:|:---:|---|
| EC-full | 前 24 epochs | 前 24 epochs | MAL (`α=0.75, γ=1.5`) | 开 | 开 | 开 | `..._ec_full_ignore9` |
| no-Mosaic + no-MAL | 关 | 前 24 epochs | focal (`α=0.25, γ=2.0`) | 开 | 开 | 开 | `..._no_mosaic_no_mal_ignore9` |

消融配置：[ecdet_l_dinov2s_patch16_dec3_liver_no_mosaic_no_mal_ignore9.yml](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_mosaic_no_mal_ignore9.yml)

### 4. 实验结果与分析

| 实验 | Model Params | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|---:|
| EC-full | 31,364,873 | 0.594 | 0.593 | 0.281 | 0.598 | 0.590 |
| no-Mosaic + no-MAL | 31,364,873 | 0.587 | 0.582 | 0.263 | 0.605 | 0.570 |

#### AP50 训练曲线

![[assets/no_mosaic_no_mal_map50_training_curve.png]]

当前训练日志中，EC-full 的最高 AP50 为 `0.5849`（epoch 46），`no-Mosaic + no-MAL` 的最高 AP50 为 `0.5858`（epoch 51）。曲线使用逐 epoch 原始值且未平滑；组合消融仍在运行，因此该图仅表示当前训练过程，不替代最终 `best.pth` 的独立评估结果。

当前 `best.pth` 的独立评估结果显示，组合移除 Mosaic 和 MAL 后，F1、mAP50 和 mAP50-95 相对 EC-full 分别下降 `0.007`、`0.011` 和 `0.018`；precision 提高 `0.007`，recall 下降 `0.020`。两者参数量相同，差异来自训练增强和分类监督，而不是模型规模。

mAP50-95 的下降大于 mAP50，说明该组合消融对高 IoU 定位质量的影响更明显。训练过程中的 AP50 峰值接近，但独立评估结果仍有差距，也说明逐 epoch 峰值不能代替最终 checkpoint 的独立评估。评估日志保存在 `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_mosaic_no_mal_ignore9/test_strict9/eval.log`。

> [!caution] 结论边界
> 该实验同时移除了 Mosaic 和 MAL，只能衡量二者的组合贡献，不能判断下降主要来自哪一项。旧 `no_mosaic` 实验仍使用 MAL，不属于本组合消融结果；若需要独立归因，还需要补充仅关闭 Mosaic 和仅替换 MAL 的两个对照。

## EC 使用 RF-DETR neck 消融

### 1. 实验目的

比较 EC-full 的三尺度 EC projection + HybridEncoder 特征路径与 RF-DETR 风格单 P4 neck 接入 EC decoder 后的检测性能和模型规模，评估 RF P4 特征路径能否以更少参数维持检测精度。

### 2. 算法实现细节

#### EC-full 路径

```text
DINOv2-S block 11/12
  → EC projection
  → P3/P4/P5（stride 8/16/32）
  → HybridEncoder（AIFI + FPN/PAN）
  → 3-level ECTransformer
```

#### RF P4 路径

```text
DINOv2-S block 3/6/9/12
  → RFMultiScaleProjector（scale=[1.0], 3 个 RF C2f blocks）
  → 单 P4（256 × 40 × 40，stride 16）
  → IdentityEncoder
  → single-level ECTransformer
```

decoder 仍为 3 层 EC/D-FINE decoder，但 feature levels 从 3 改为 1，deformable sampling points 从 `[3,6,3]` 改为 `[2]`。MAL、FDR、FGL、DDF、GO union、CDN、训练计划和严格 9 类数据口径保持不变。

> [!warning] 消融边界
> 该严格 RF P4 实验同时替换 projector、删除 HybridEncoder，并把 decoder 输入从三尺度改为单尺度。因此它衡量的是完整 RF P4 特征路径，不是只替换一个 neck module 的纯单变量实验。

### 3. 实验具体配置

| 项目 | EC-full | RF P4 neck |
|---|---|---|
| DINOv2 blocks | `[10,11]`（zero-based） | `[2,5,8,11]`（zero-based） |
| Projector | EC projection | RFMultiScaleProjector，3 RF C2f blocks |
| 输出尺度 | P3/P4/P5 | P4 only |
| Feature strides | `[8,16,32]` | `[16]` |
| Encoder | HybridEncoder | IdentityEncoder |
| Decoder levels | 3 | 1 |
| Sampling points | `[3,6,3]` | `[2]` |
| Decoder layers | 3 | 3 |
| 参数量 | 31,364,873 | 27,757,881 |
| 输出目录 | `..._ec_full_ignore9` | `..._rf_neck_p4_ignore9` |

RF P4 配置：[ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_ignore9.yml](../ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_ignore9.yml)

### 4. 实验结果与分析

| 实验 | Model Params | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|---:|
| EC-full | 31,364,873 | 0.594 | 0.593 | 0.281 | 0.598 | 0.590 |
| RF P4 neck | 27,757,881 | 0.589 | 0.587 | 0.274 | 0.554 | 0.630 |

RF P4 路径减少 3,606,992 个参数，约为 EC-full 的 `11.5%`，mAP50-95 下降 `0.007`。mAP50 和 F1 小幅下降，而 recall 从 `0.590` 提高到 `0.630`、precision 从 `0.598` 降到 `0.554`，表现为召回略高、定位和分类精度略低。

当前结果说明单 P4 RF 路径在明显缩减参数的情况下保持了接近 EC-full 的 mAP50-95。但由于该配置同时移除了 HybridEncoder 和多尺度输入，不能把 `-0.007` 直接解释为 RF projector 本身的贡献。若要做严格 neck-only 结论，需要增加“RF 三尺度 projector + 原 HybridEncoder”的严格 9 类对照。

## 全部实验结果汇总

| 实验                 | Model Params | Best Epoch |        F1 |     mAP50 |  mAP50-95 |         P |         R |
| ------------------ | -----------: | ---------: | --------: | --------: | --------: | --------: | --------: |
| EC-full            |   31,364,873 |         46 |     0.594 |     0.593 |     0.281 |     0.598 |     0.590 |
| no-CDN             |   31,362,313 |         53 |     0.594 |     0.589 |     0.270 |     0.570 |     0.620 |
| no-FDR decode      |   31,761,300 |         45 | **0.598** | **0.601** | **0.283** |     0.569 | **0.630** |
| no-GO-DDF          |   31,364,873 |         46 |     0.592 |     0.592 |     0.282 |     0.574 |     0.610 |
| no-FDR + no-GO-DDF |   31,264,776 |         47 |     0.588 |     0.588 |     0.278 |     0.559 |     0.620 |
| no-Mosaic + no-MAL |   31,364,873 |         47 |     0.587 |     0.582 |     0.263 | **0.605** |     0.570 |
| RF P4 neck         |   27,757,881 |         54 |     0.589 |     0.587 |     0.274 |     0.554 | **0.630** |

`Best Epoch` 是训练过程中 mAP50-95 最高、用于生成 `best.pth` 的 epoch；表中其余指标来自该 checkpoint 的独立评估。

### 主要结论

1. D-FINE 三组消融的 mAP50-95 相对 EC-full 仅变化 `+0.002`、`+0.001` 和 `-0.003`，当前单 seed 下影响有限，不能证明这些机制有稳定增益。
2. `no-FDR decode` 仍保留 FGL/DDF 分布监督，并额外使用连续回归 head，因此它的小幅提升不能解释为“FDR 分布无效”。
3. `no-Mosaic + no-MAL` 下降最大，mAP50-95 降低 `0.018`，说明训练增强和分类监督的组合影响大于本轮 D-FINE 机制消融。
4. RF P4 neck 减少约 `11.5%` 参数，mAP50-95 仅下降 `0.007`，是当前最明显的参数效率改进，但 precision 有所下降。
5. CDN 的收益也较有限：关闭后 F1 不变、mAP50-95 下降 `0.011`，主要改变 precision/recall 平衡，而不是决定模型主体性能。

## 结果更新规则

1. 只在任务完成并生成 `best.pth` 后填写结果。
2. `mAP50` 和 `mAP50-95` 直接读取 COCO evaluator 输出。
3. F1 是 IoU50 PR 曲线上的最大 macro-F1；R 是该工作点的 recall-grid。
4. P 由同一工作点的 `P = F1 × R / (2R − F1)` 计算。
5. 旧 13 类 checkpoint 的 evaluator-only ignore 结果不得填入严格 9 类表格。
