---
title: "EdgeCrafter 严格 9 类消融实验报告"
status: "active"
cluster: "EdgeCrafter 病灶检测"
tags:
  - type/experiment-report
  - domain/computer-vision
  - domain/medical-imaging
  - topic/object-detection
  - topic/ablation
  - topic/d-fine
  - topic/denoising
  - project/edgecrafter
aliases:
  - EC_ABLATION_REPORT
  - EC 严格 9 类报告
updated: "2026-08-07"
source: "/cobot/Code/wanrui/EdgeCrafter/docs/EC_ABLATION_REPORT.md"
source_updated: "2026-08-07 11:26 CST"
cssclasses:
  - wide-page
summary: "记录 EdgeCrafter 严格 9 类训练任务的 valid 指标、消融定义、运行状态与归因边界。"
---

# EdgeCrafter 严格 9 类消融实验报告

> [!summary] 当前结论
> - 严格 9 类 EC-full 的 valid AP 为 **0.269840**。
> - 关闭 CDN 后 valid AP 为 **0.273587（+0.003746）**；差异较小，目前只说明 CDN 没有呈现正向增益。
> - RF P4 路径将参数量减少 **3.61M（约 11.5%）**，valid AP 仅下降 **0.001371**，但它同时替换 projector、移除 HybridEncoder 并改为单尺度，不能归因给单个 neck module。
> - D-FINE 三组以及 no-Mosaic + no-MAL 仍在训练，临时 best 值不进入正式结果表。

> [!important] 结果口径
> 本文只记录使用严格 9 类 train/valid 标注训练的模型，并只报告 `valid_ignore_9_12.json` 指标。Best epoch 按 valid COCO AP@[0.50:0.95] 选择；test 结果单独保存，不混入本文。

相关笔记：[[20-Projects/EdgeCrafter/EC_ABLATIONS|EC 消融设计与历史结果]] · [[30-Knowledge/Computer-Vision/计算机视觉评价指标|计算机视觉评价指标]] · [[30-Knowledge/Algorithms/匈牙利匹配|匈牙利匹配]]

## 导航

- [[#1. 统一实验口径|统一实验口径]]
- [[#2. 当前结果与任务状态|当前结果与任务状态]]
- [[#3. CDN 消融|CDN 消融]]
- [[#4. D-FINE decoder 消融|D-FINE decoder 消融]]
- [[#5. No-Mosaic + No-MAL 组合消融|No-Mosaic + No-MAL]]
- [[#6. RF P4 特征路径消融|RF P4 特征路径]]
- [[#7. 结果更新规则|结果更新规则]]

---

## 1. 统一实验口径

所有实验以严格 9 类 EC-full 为对照，除各节列出的变量外，其余设置保持一致。

| 项目 | 统一配置 |
| --- | --- |
| 数据集 | `det_liver` |
| 类别 | `num_classes=9`，category ID 0–8 |
| Train 标注 | `/cobot/Data/Lesion_det/det_liver/annotations/train_ignore_9_12.json` |
| Valid 标注 | `/cobot/Data/Lesion_det/det_liver/annotations/valid_ignore_9_12.json` |
| 输入尺寸 | 640 × 640 |
| Backbone | 无 register DINOv2-S；12 blocks、384 dim、6 heads、patch16 |
| EC-full 路径 | EC 三尺度 projection → HybridEncoder → ECTransformer |
| Decoder | 3 层，300 normal queries |
| 优化器 | AdamW；backbone LR `5e-6`，其余 LR `5e-4` |
| 训练计划 | 150 epochs；warmup 2,000 iterations；patience 30 |
| Batch | 总 batch size 32，4 GPU |
| 训练设置 | seed 42、AMP、SyncBN、EMA |
| 分类损失 | MAL：`alpha=0.75`，`gamma=1.5` |
| 定位路径 | FDR + L1 + GIoU + FGL + DDF + GO union |
| CDN | `num_denoising=100` |
| 增强 | Mosaic 与 MixUp 前 24 epochs 开启；最后 2 epochs 关闭强增强 |

基线配置：

```text
/cobot/Code/wanrui/EdgeCrafter/ecdetseg/configs/ecdet/
  ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml
```

> [!warning] 与历史结果的区别
> [[20-Projects/EdgeCrafter/EC_ABLATIONS|EC_ABLATIONS]] 中的旧表主要是“13 类训练 checkpoint + evaluator-only ignore 9–12”。本文是从训练开始就使用严格 9 类标注，两者不能直接做单变量归因。

---

## 2. 当前结果与任务状态

### 2.1 已完成结果

| 实验 | Params | Best epoch | Valid AP | AP50 | AP75 | AR@100 | Macro-F1 | ΔAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **EC-full strict-9** | 31,364,873 | 46 | 0.269840 | 0.585 | **0.214** | 0.519 | 0.594197 | — |
| No-CDN strict-9 | 31,362,313 | 53 | **0.273587** | **0.595** | 0.201 | 0.527 | **0.600061** | **+0.003746** |
| RF P4 neck strict-9 | 27,757,881 | 54 | 0.268470 | 0.578 | 0.206 | **0.533** | 0.590744 | -0.001371 |

### 2.2 当前任务状态

状态快照：2026-08-07。

| 实验 | Slurm job | 状态 |
| --- | ---: | --- |
| EC-full strict-9 | 直接 GPU 运行 | 已完成 valid / test |
| No-CDN strict-9 | 749 | 已完成 valid / test |
| No-FDR decode strict-9 | 752 | 训练中 |
| No-GO-DDF strict-9 | 750 | 训练中 |
| No-FDR + No-GO-DDF strict-9 | 751 | 训练中 |
| No-Mosaic + No-MAL strict-9 | 756 | 训练中 |
| RF P4 neck strict-9 | 直接 GPU 运行 | 已完成 valid / test |

> [!note] 状态维护
> 本表是文档同步时的快照，不作为实时 GPU 看板。结果只有在任务完成并生成 `best.pth` 后才能进入正式表。

---

## 3. CDN 消融

### 3.1 研究问题

CDN 在训练期加入带标签噪声和框噪声的 DN queries，但不改变最终 300 个 normal queries 的推理输出。本实验测试它对收敛和 valid 指标的独立贡献。

### 3.2 实现差异

EC-full：

```yaml
ECTransformer:
  num_denoising: 100
```

No-CDN：

```yaml
ECTransformer:
  num_denoising: 0
```

关闭 CDN 后：

- 不创建 `denoising_class_embed`；
- 不调用 `get_contrastive_denoising_training_group()`；
- 不拼接 DN queries 或构造 DN attention mask；
- model output 不再包含 `dn_outputs`、`dn_meta`；
- criterion 不产生 `loss_*_dn_*`；
- normal queries、MAL、GO union、FDR、FGL、DDF 和推理路径保持不变。

### 3.3 配置与结果

| 实验 | `num_denoising` | Params | 输出目录后缀 |
| --- | ---: | ---: | --- |
| EC-full | 100 | 31,364,873 | `..._ec_full_ignore9` |
| No-CDN | 0 | 31,362,313 | `..._no_cdn_ignore9` |

No-CDN 配置：

```text
ecdetseg/configs/ecdet/
  ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_ignore9.yml
```

> [!conclusion] 当前解释
> 关闭 CDN 后 AP 提高约 0.37 个百分点，AP50、AR@100 和 macro-F1 略升，但 AP75 下降 1.3 个百分点。CDN 只减少 2,560 个参数，说明变化来自训练 query/loss，而不是模型规模。单 seed 下不能判断这是稳定收益。

实现入口：

- `ecdetseg/engine/edgecrafter/decoder.py`
- `ecdetseg/engine/edgecrafter/denoising.py`
- `ecdetseg/engine/edgecrafter/criterion.py`

---

## 4. D-FINE decoder 消融

### 4.1 研究问题

拆分三个相关但不等价的机制：

1. FDR 分布式边界解码；
2. GO union matching；
3. FGL / DDF 分布监督。

### 4.2 机制边界

#### FDR

EC-full 的每层 decoder 输出四条离散边界分布 `pred_corners`，通过 Integral/FDR 转为 box，并由 LQE 使用分布质量修正分类分数。

`no-FDR decode`：

- 设置 `use_fdr_decode=false`；
- 最终框由独立连续 4 维 box head 生成；
- 保留 `use_aux_distribution=true`；
- 继续计算 FGL 和 DDF。

因此它是最终解码路径对照，不是完整移除 D-FINE。

#### GO union 与 DDF

EC-full 分别对 final、auxiliary、pre-output 和 encoder output 做 Hungarian matching，再由 `_get_go_indices()` 合并为定位损失使用的 union set。

- `use_uni_set=false`：各层使用自己的 local matching；
- `use_ddf=false`：不再产生 `loss_ddf*`；
- `use_fgl=true`：仍保留 GT 边界分布监督。

#### Continuous-box-only 组合

`no-FDR + no-GO-DDF` 同时关闭：

- FDR 最终框解码；
- distribution auxiliary output；
- LQE；
- FGL；
- DDF；
- GO union。

Criterion 只保留 `['mal', 'boxes']`。这一组才是当前矩阵中的 continuous-box-only 对照。

### 4.3 实验矩阵

| 实验 | FDR | 分布辅助 | LQE | GO | FGL | DDF | 分类 | 状态 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | --- | --- |
| EC-full | 开 | 开 | 开 | 开 | 开 | 开 | MAL | 已完成 |
| No-FDR decode | 关 | 开 | 关 | 开 | 开 | 开 | MAL | 训练中 |
| No-GO-DDF | 开 | 开 | 开 | 关 | 开 | 关 | MAL | 训练中 |
| No-FDR + No-GO-DDF | 关 | 关 | 关 | 关 | 关 | 关 | MAL | 训练中 |

配置：

- `..._no_fdr_decode_ignore9.yml`
- `..._no_go_ddf_ignore9.yml`
- `..._no_fdr_no_go_ddf_ignore9.yml`

> [!note] 分析状态
> 三组任务尚未完成，结果单元格保持空白。训练中的临时 best 值不能作为最终 valid 结果。

---

## 5. No-Mosaic + No-MAL 组合消融

### 5.1 研究问题

同时移除 Mosaic 数据增强和 MAL 分类目标，测试两项策略共同移除后的组合影响。该实验不能分别归因给 Mosaic 或 MAL。

### 5.2 关闭 Mosaic

`mosaic_prob=0.0` 后，Mosaic transform 仍在 Compose 中，但调度条件恒为 false。

仍然保留：

- 前 24 epochs 的 MixUp（`mixup_prob=1.0`）；
- RandomPhotometricDistort；
- RandomHorizontalFlip；
- RandomZoomOut 和 RandomIoUCrop；
- 原训练计划和最后 2 epochs 的 no-augmentation 设置。

### 5.3 MAL 替换为 Focal

MAL 使用匹配框 IoU 的 `gamma` 次幂作为正样本软目标，并用 detach 后的预测分数构造负样本权重。关闭 MAL 时不能删除分类监督，因此改为 sigmoid focal loss：

```yaml
ECCriterion:
  losses: ['focal', 'boxes', 'local']
  alpha: 0.25
  gamma: 2.0
```

`loss_focal` 权重为 1；L1、GIoU、FGL、DDF、GO union 和 CDN 保持开启。

### 5.4 实验矩阵

| 实验 | Mosaic | MixUp | 分类损失 | FGL/DDF | GO | CDN | 状态 |
| --- | :---: | :---: | --- | :---: | :---: | :---: | --- |
| EC-full | 前 24 epochs | 前 24 epochs | MAL（α=0.75，γ=1.5） | 开 | 开 | 开 | 已完成 |
| No-Mosaic + No-MAL | 关 | 前 24 epochs | Focal（α=0.25，γ=2.0） | 开 | 开 | 开 | 训练中 |

配置：

```text
ecdet_l_dinov2s_patch16_dec3_liver_no_mosaic_no_mal_ignore9.yml
```

> [!warning] 归因边界
> 旧 `no_mosaic` 实验仍使用 MAL，不属于本组合消融；本任务也不能单独回答 Mosaic 或 MAL 各自的贡献。

---

## 6. RF P4 特征路径消融

### 6.1 研究问题

比较 EC-full 的三尺度 EC projection + HybridEncoder，与 RF-DETR 风格的单 P4 特征路径，观察是否能用更少参数维持 valid 性能。

### 6.2 两条特征路径

EC-full：

```text
DINOv2-S block 11/12
  -> EC projection
  -> P3/P4/P5（stride 8/16/32）
  -> HybridEncoder（AIFI + FPN/PAN）
  -> 3-level ECTransformer
```

RF P4：

```text
DINOv2-S block 3/6/9/12
  -> RFMultiScaleProjector（scale=[1.0]，3 RF C2f blocks）
  -> P4（256 × 40 × 40，stride 16）
  -> IdentityEncoder
  -> single-level ECTransformer
```

Decoder 仍是 3 层 EC/D-FINE decoder，但 feature levels 从 3 改为 1，sampling points 从 `[3,6,3]` 改为 `[2]`。MAL、FDR、FGL、DDF、GO、CDN 和训练计划保持不变。

### 6.3 配置对照

| 项目 | EC-full | RF P4 |
| --- | --- | --- |
| DINOv2 blocks | `[10,11]` | `[2,5,8,11]` |
| Projector | EC projection | RFMultiScaleProjector |
| 输出 | P3 / P4 / P5 | P4 only |
| Strides | `[8,16,32]` | `[16]` |
| Encoder | HybridEncoder | IdentityEncoder |
| Decoder levels | 3 | 1 |
| Sampling points | `[3,6,3]` | `[2]` |
| Decoder layers | 3 | 3 |
| Params | 31,364,873 | 27,757,881 |

RF P4 配置：

```text
ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_ignore9.yml
```

### 6.4 结果解释

- 参数减少：3,606,992（约 11.5%）；
- Valid AP：0.269840 → 0.268470（-0.001371）；
- AR@100：0.519 → 0.533；
- AP50、AP75 和 macro-F1 略降。

> [!conclusion] 当前解释
> 单 P4 RF 路径在明显缩减参数的同时保持了接近 EC-full 的 valid AP，呈现“召回略高、定位与分类精度略低”的趋势。

> [!warning] 消融边界
> 该实验同时替换 projector、删除 HybridEncoder，并把 decoder 输入从三尺度改为单尺度。它衡量的是完整 RF P4 特征路径，不是纯粹的 neck-only 单变量。严格 neck 归因还需要“RF 三尺度 projector + 原 HybridEncoder”的严格 9 类对照。

---

## 7. 结果更新规则

1. 只在任务完成并生成 `best.pth` 后填写结果。
2. 使用 `valid_ignore_9_12.json` 重评 `best.pth`，或读取 best epoch 当次 valid 输出。
3. Best epoch 由 valid COCO AP@[0.50:0.95] 决定。
4. F1、AP50、AP75 和 AR 必须与 AP 来自同一个 best epoch。
5. Test 指标单独记录，不覆盖本文 valid 表格。
6. 旧 13 类 checkpoint 的 evaluator-only ignore 结果不得填入严格 9 类表格。
7. 每次从 SSH 更新时先比较差异，再修改 `source_updated` 和状态快照日期。

> [!info] 同步说明
> 远端原始报告位于 `/cobot/Code/wanrui/EdgeCrafter/docs/EC_ABLATION_REPORT.md`。本文件是 Obsidian 阅读版；代码链接统一保留为 SSH 仓库相对路径，避免生成无法在本机打开的 Markdown 链接。
