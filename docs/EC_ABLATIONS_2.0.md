# EdgeCrafter 消融 2.0

本文记录当前 EdgeCrafter 检测分支中 CDN、FDR、FGL、DDF 和 GO union 的实现边界，以及 D-FINE 正式消融链、Continuous + GO 辅助实验和历史诊断配置的真实实现边界。本文以仓库代码为准，主要对应 strict-9 liver 配置：

```text
train_ignore_9_12.json
valid_ignore_9_12.json
test_ignore_9_12.json
```

代码中使用的术语是 **FDR**（Fine-grained Distribution Refinement）。如果其他材料中写作 FDL，本文统一按当前代码的 FDR 命名。

## 零、已有实验结果总览（更新于 2026-08-13）

本节汇总当前已完成的 decoder depth、CDN、D-FINE、MAL/Focal 和 neck/interface 实验。除明确写出的目标变量外，实验均使用 strict-9 liver 数据、DINOv2-S no-register backbone、全局 batch size 32、2 GPU、seed 42、AMP 和 EMA；验证指标由 EMA 模型计算。

结果统一采用以下口径：

- `Best Epoch` 是 `log.txt` 中从 0 开始记录的 epoch，也是 `best.pth` 中的 `last_epoch`；例如 `43` 表示完成第 44 轮训练。
- `best.pth` 只按 COCO `mAP50-95` 选择；表中的 F1、mAP50、P、R 均来自该 `best.pth` 对应的同一 epoch，不拼接其他 epoch 的历史最高值。
- F1 是 IoU=0.50 时 COCO PR 曲线上的最大 macro-F1。R 是最大 F1 所在的 recall grid，P 是同一工作点的 macro precision，因此它们不是固定置信度阈值下的 P/R。
- mAP50 和 mAP50-95 使用 `pycocotools.COCOeval`，以下指标统一写成百分数。

### 0.1 Decoder depth 消融：3/4/5 层

这三组是同一批 decoder 重测，只改变 `ECTransformer.num_layers`，其余数据、模型开关、增强、优化器和训练计划保持一致。

| 实验 | Params | Best Epoch | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|
| EC-full，3-layer decoder | 31.365M | 46 | 59.64 | 58.90 | 27.54 | 58.34 | 61.00 |
| EC-full，4-layer decoder | 32.657M | 43 | 60.25 | 59.93 | 27.98 | 59.51 | 61.00 |
| EC-full，5-layer decoder | 33.950M | 43 | **60.64** | **60.16** | **28.25** | **61.30** | 60.00 |

相对 3 层，4 层增加 1.292M 参数，F1、mAP50、mAP50-95 分别提高 0.60、1.03、0.44 个百分点；5 层增加 2.585M 参数，分别提高 1.00、1.27、0.71 个百分点。5 层在三个主指标上均为当前最佳，但 4→5 层的增益已经小于 3→4 层，呈现边际收益递减。

5 层在 epoch 45 的历史最高 F1 为 61.03%，但该轮 mAP50-95 为 28.18%，低于 epoch 43 的 28.25%；因此正式表仍使用由 mAP50-95 选出的 epoch 43。

对应输出目录：

```text
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_baseline_nofile65536_2gpu_seed42
outputs/ablation/ecdet_l_dinov2s_patch16_dec4_liver_baseline_nofile65536_2gpu_seed42
outputs/ablation/ecdet_l_dinov2s_patch16_dec5_liver_baseline_nofile65536_2gpu_seed42
```

#### 0.1.1 Decoder depth × D-FINE 端点对照

为检验 D-FINE 是否依赖更深的逐层 refinement，在已有 3-layer 与 5-layer EC-full 上，分别与 strict no-D-FINE 进行端点对照。no-D-FINE 删除 distribution/Integral/LQE/pre/FGL/DDF/GO 路径，保留连续框回归、MAL、CDN 和其余训练协议。

| Decoder depth | D-FINE | Params | Best Epoch | F1 | mAP50 | mAP50-95 | P | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 开 | 31.365M | 49 | 59.10 | 58.06 | 27.36 | 58.23 | 60.00 |
| 3 | 关 | 31.132M | 40 | **60.61** | **59.98** | **27.69** | **58.40** | **63.00** |
| 5 | 开 | 33.950M | 43 | 60.64 | **60.16** | **28.25** | **61.30** | 60.00 |
| 5 | 关 | 33.651M | 47 | **60.80** | 59.56 | 27.64 | 59.65 | **62.00** |

在 3-layer 下，移除 D-FINE 后 mAP50-95 反而提高 0.33 个百分点；在 5-layer 下，移除 D-FINE 后 mAP50-95 下降 0.61 个百分点、mAP50 下降 0.60 个百分点。两个端点的 D-FINE 效应相差 0.94 个百分点，形成一个比较明确的深度交互信号：D-FINE 的分布累积与深层到浅层自蒸馏，可能需要足够的 decoder 迭代深度才能转化为更严格 IoU 下的定位收益。

不过，5-layer no-D-FINE 与早先的 5-layer EC-full 虽然协议相同，仍是独立训练运行；当前只有 seed 42，因此更稳妥的结论是“观察到 decoder depth × D-FINE 交互倾向”，而不是已经证明两者必然互补。

新增输出目录：

```text
outputs/ablation/ecdet_l_dinov2s_patch16_dec5_liver_no_dfine_ignore9_bs32_2gpu_seed42
```

### 0.2 CDN 消融

CDN 消融使用同批核心消融中的 EC-full 作为对照。`EC - CDN` 只设置 `num_denoising: 0`，D-FINE、MAL、普通 300 queries、backbone、neck 和三层 decoder 均保持不变。

| 实验 | CDN | Params | Best Epoch | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EC-full | 开 | 31.365M | 49 | 59.10 | 58.06 | 27.36 | 58.23 | 60.00 |
| EC - CDN | 关 | 31.362M | 49 | **60.32** | **59.20** | **27.57** | 57.86 | **63.00** |

单 seed 下，关闭 CDN 后 F1、mAP50、mAP50-95 分别提高 1.22、1.14、0.22 个百分点，参数仅减少 2,560 个，即 `Embedding(num_classes + 1, hidden_dim)`。结果说明本配置下 CDN 没有呈现正收益，主要差异来自训练监督与 query 构造，而不是模型容量；但 mAP50-95 差异只有 0.22 个百分点，仍需多 seed 验证。

对应输出目录：

```text
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9_nofile65536_bs32_2gpu_seed42
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_ignore9_nofile65536_bs32_2gpu_seed42
```

#### 0.2.1 D-FINE × CDN 完整 2×2

新增 `EC - D-FINE - CDN` 后，D-FINE 与 CDN 的四个角点已齐全。该新实验以 strict `EC - D-FINE` 为对照，只将 `num_denoising` 从 100 设为 0。

| D-FINE | CDN | Params | Best Epoch | F1 | mAP50 | mAP50-95 | P | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 开 | 开 | 31.365M | 49 | 59.10 | 58.06 | 27.36 | **58.23** | 60.00 |
| 开 | 关 | 31.362M | 49 | **60.32** | **59.20** | **27.57** | 57.86 | **63.00** |
| 关 | 开 | 31.132M | 40 | **60.61** | **59.98** | **27.69** | 58.40 | **63.00** |
| 关 | 关 | 31.130M | 55 | 60.14 | 59.61 | 27.32 | **60.28** | 60.00 |

CDN 的效应随 D-FINE 状态发生反转：

```text
D-FINE 开：CDN on - CDN off = 27.36 - 27.57 = -0.21
D-FINE 关：CDN on - CDN off = 27.69 - 27.32 = +0.37
两种条件下的 CDN effect 相差 0.58 个百分点
```

在连续回归路径中，保留 CDN 使 F1、mAP50、mAP50-95 分别提高 0.47、0.37、0.37 个百分点，R 提高 3 个百分点，但 P 降低 1.88 个百分点。这表明 CDN 本身不能简单判定为“无效”：它在 no-D-FINE 下有正向召回与 AP 收益，先前 EC-full 中的轻微负收益更可能是 CDN 与 distribution/pre/DDF 训练路径的交互，而不是 CDN 作为独立机制始终为负。

该交互幅度仍小于 1 个百分点，且只有单 seed，应表述为“交互迹象”，不应直接上升为稳定机制结论。

新增输出目录：

```text
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_dfine_no_cdn_ignore9_bs32_2gpu_seed42
```

### 0.3 D-FINE 四组消融

四组实验构成当前正式 D-FINE 主链：EC-full、EC - GO-LSD、EC - D-FINE，以及用于诊断连续回归下 GO 效果的 Continuous + GO。`EC - GO-LSD` 的运行别名/输出目录使用 `no_go_lsd_ignore9`，实际配置文件仍为 `no_go_ddf_ignore9.yml`。

| 实验 | FDR family | GO | LSD/DDF | Params | Best Epoch | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EC-full | 开 | 开 | 开 | 31.365M | 49 | 59.10 | 58.06 | 27.36 | 58.23 | 60.00 |
| EC - GO-LSD | 开 | 关 | 关 | 31.365M | 45 | 60.42 | 59.32 | 27.98 | 58.92 | 62.00 |
| EC - D-FINE | 关 | 关 | 不适用 | 31.132M | 40 | 60.61 | 59.98 | 27.69 | 58.40 | 63.00 |
| Continuous + GO | 关 | 开 | 不适用 | 31.132M | 42 | **60.75** | **60.85** | **28.19** | 58.65 | **63.00** |

按照第 2.6 节定义的归因方向，单 seed 的 mAP50-95 差值为：

```text
FDR contribution
= (EC - GO-LSD) - (EC - D-FINE)
= 27.98 - 27.69
= +0.29

GO-LSD contribution
= EC-full - (EC - GO-LSD)
= 27.36 - 27.98
= -0.62

full D-FINE contribution
= EC-full - (EC - D-FINE)
= 27.36 - 27.69
= -0.33

continuous-regression GO contribution
= Continuous + GO - (EC - D-FINE)
= 28.19 - 27.69
= +0.50
```

当前结果显示：保留 FDR、关闭 GO-LSD 后取得 27.98%，比 EC-full 高 0.62 个百分点；完全移除 D-FINE 后为 27.69%，与 EC-full 接近；在连续回归基线上重新打开 GO 后达到 28.19%，为四组最高。这说明 GO 在连续回归路径中可能有正作用，而完整 GO-LSD 与当前 FDR 训练组合并未在该 seed 上形成正增益。

上述差异仍不能直接作为稳定的模块结论。相同 EC-full 配置的独立 decoder-3 重测得到 27.54%，与核心 EC-full 的 27.36% 相差 0.18 个百分点；部分 D-FINE 差值与这一运行波动处于相近量级。正式报告应补充至少 3 个 seed，报告均值和标准差。

对应输出目录：

```text
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9_nofile65536_bs32_2gpu_seed42
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_go_lsd_ignore9_nofile65536_bs32_2gpu_seed42
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_dfine_ignore9_nofile65536_bs32_2gpu_seed42
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_continuous_with_go_ignore9_nofile65536_bs32_2gpu_seed42
```

### 0.4 结果使用注意事项

- decoder depth 表中的 3 层是独立重测；CDN 与 D-FINE 表使用核心消融批次的 EC-full。两次 EC-full 配置相同但随机训练过程独立，不能把其中一条替换到另一张表后计算差值。
- 所有结果均启用 EMA，验证和 `best.pth` 选择使用 EMA 权重；checkpoint 同时保存即时模型与 EMA 状态。
- 部分 runner 在训练和 checkpoint 保存完成后出现 shell 尾部引号错误。该错误不影响上述 `log.txt`、`best.pth` 或训练期验证结果，但会阻断 runner 的后处理收尾，应在后续批量实验前修复。

### 0.5 Mosaic × 分类损失消融

当前 strict-9 下已有 `Mosaic + MAL`、`Mosaic + Focal` 和 `No Mosaic + Focal` 三个角点。其中 “MAL-off” 不是删除分类监督，而是将 MAL 替换为标准 sigmoid Focal Loss；框仍由原 EC/D-FINE 定位路径预测，不因分类 loss 替换而改成另一种框表示。

Focal 配置为：

```yaml
ECCriterion:
  weight_dict: {loss_focal: 1, loss_bbox: 5, loss_giou: 2, loss_fgl: 0.15, loss_ddf: 1.5}
  losses: ['focal', 'boxes', 'local']
  alpha: 0.25
  gamma: 2.0
```

`Mosaic + Focal` 保留 Mosaic，MixUp、其余数据增强、D-FINE、CDN、backbone、neck 和三层 decoder 均保持同批 EC-full 设置，因此可以直接隔离 MAL/Focal 的影响。

| 实验 | Mosaic | 分类损失 | Params | Best Epoch | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| EC-full paired baseline | 开 | MAL | 31.365M | 44 | 59.54 | **59.39** | **27.79** | **59.09** | 60.00 |
| Mosaic + Focal | 开 | Focal | 31.365M | 47 | **60.03** | 58.83 | 26.04 | 57.33 | **63.00** |

在 Mosaic 保持开启时，Focal 相对 MAL 的变化为：

```text
F1       +0.49
mAP50    -0.56
mAP50-95 -1.75
P        -1.76
R        +3.00
```

这是目前五项新结果中最清晰的负向消融之一。Focal 将最佳 F1 工作点推向更高召回，但牺牲 precision，同时 mAP50-95 明显下降。由于 box/FDR 结构未改，这不是“Focal 把离散框换成连续框”；更合理的解释是，分类监督改变了 query 的排序、正负样本梯度和分类与定位质量的对齐，最终影响 COCO AP 排序及高 IoU 框的保留。就当前 seed 而言，MAL 更适合作为正式基线的分类监督。

`No Mosaic + Focal` 作为第三个角点，完整结果为：

| 实验 | Mosaic | 分类损失 | Params | Best Epoch | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| EC - Mosaic - MAL | 关 | Focal | 31.365M | 42 | 60.00 | 58.90 | 26.33 | 62.15 | 58.00 |

它与本次 `Mosaic + Focal` 不是同批配对运行，因此 0.29 个百分点的 mAP50-95 差异只能作趋势参考，不宜用来独立宣称 Mosaic 的主效应。要完成严格 2×2，仍需要一组与 EC-full 同口径的 `No Mosaic + MAL`。

对应输出目录：

```text
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_neck_pair_ignore9_bs32_2gpu_seed42
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_mosaic_focal_ignore9_bs32_2gpu_seed42
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_mosaic_no_mal_ignore9_nofile65536_bs32_2gpu_seed42
```

### 0.6 多组块消融的架构化组合建议

#### 0.6.1 先按作用位置分块

建议将变量按数据流中的位置分成五块，而不是把所有开关做全排列：

| 块 | 变量 | 主要作用位置 | 典型问题 |
|---|---|---|---|
| A：输入与增强 | Mosaic（必要时另测 MixUp） | 图像和 GT 进入 backbone 之前 | 输入分布、尺度/布局多样性是否有效 |
| B：分类监督 | MAL / Focal | matcher 之后的分类梯度与分数质量 | IoU-aware soft target 是否优于硬 one-hot target |
| C：定位算法 | FDR、GO-LSD、Continuous + GO | decoder box 更新、跨层匹配和分布蒸馏 | D-FINE 的哪些定位机制有效 |
| D：解码容量与训练 query | decoder 3/4/5 层、CDN on/off | decoder 深度和训练期 query 构造 | 更深 refinement 与 denoising 是否互补 |
| E：特征接口与 neck | EC projector + HybridEncoder + P3/P4/P5 / strict RF projector + IdentityEncoder + P4 | backbone stage 输出到 decoder memory 之间 | EC 多尺度 neck 是否优于 RF 的单 P4 projector；单尺度 decoder 的采样密度是否敏感 |

这种分块的原因是：Mosaic 不改变模型结构；MAL/Focal 改变分类监督；D-FINE 改变定位表示与匹配；decoder depth/CDN 改变迭代容量和训练路径；neck 改变 decoder 接收到的特征尺度、token 数量和语义融合。跨块组合只应在有明确交互假设时增加。

#### 0.6.2 第一阶段：补齐块内最小可归因设计

##### A×B：Mosaic × 分类损失，完整 2×2

当前已有三个角点，只剩 `No Mosaic + MAL` 尚未按当前 strict-9 口径补齐：

| 编号 | Mosaic | 分类损失 | 当前状态 | 能回答的问题 |
|---|---:|---|---|---|
| AB-1 | 开 | MAL | 已有：EC-full | 基线 |
| AB-2 | 关 | MAL | **需补** | Mosaic 的主效应 |
| AB-3 | 开 | Focal | **已完成：`Mosaic + Focal Loss`** | MAL 相对 Focal 的主效应 |
| AB-4 | 关 | Focal | 已有：EC - Mosaic - MAL | 二者联合效果 |

由四组可计算：

```text
Mosaic effect under MAL   = AB-1 - AB-2
Mosaic effect under Focal = AB-3 - AB-4
MAL effect with Mosaic    = AB-1 - AB-3
MAL effect without Mosaic = AB-2 - AB-4

interaction
= (AB-1 - AB-2) - (AB-3 - AB-4)
```

如果两条 Mosaic effect 接近，则主要是可加的主效应；如果符号或幅度明显不同，说明 Mosaic 与 MAL/Focal 存在交互，联合实验不能拆开解释。

##### C：定位算法块

当前四组已经足够作为最小主链，不建议继续扩张为所有开关排列：

```text
EC-full
EC - GO-LSD
EC - D-FINE
Continuous + GO
```

DDF 依赖 FDR distribution，因此不能构造一个真正对称的 FDR×GO-LSD 2×2。若还要增加一组，优先做 `FDR + GO, no DDF`，用于把 GO 与 LSD/DDF 分开；但只有在论文需要分别声明 GO 和 LSD 贡献时才值得增加。

##### D：decoder depth

已有 3/4/5 层 EC-full 结果足以完成容量趋势分析，并已补齐 5-layer no-D-FINE 端点。5 层 EC-full 的 mAP50-95 最佳，且 D-FINE 在 5 层下呈现正收益；后续不必继续扩展到 4-layer no-D-FINE，应优先对 3/5-layer 端点补 seed。

##### E：neck 与 decoder memory 接口

neck 主效应必须与同批次 EC-full 配对。当前 strict RF-neck 定义为：DINOv2-S blocks 3/6/9/12 经 RF projector 融合为单一 P4，`IdentityEncoder` 绕过 EC HybridEncoder，然后仍送入原 EC D-FINE decoder。它只改变 neck 和 decoder 的输入接口，不是 RF decoder。

本轮已完成的最小配对如下：

| 实验 | Neck / decoder 输入 | EC deformable sampling | 状态 | 能回答的问题 |
|---|---|---:|---|---|
| E-1 | EC projector + HybridEncoder，P3/P4/P5 | `[3,6,3]`，每 head 共 12 点 | **已完成，mAP50-95 27.79** | 配对基线 |
| E-2 | strict RF projector + IdentityEncoder，P4-only | `[6]`，每 head 6 点 | **已完成，mAP50-95 27.34** | RF 单 P4 neck 在较充分 EC 采样下的效果 |
| E-3 | strict RF projector + IdentityEncoder，P4-only | `[2]`，每 head 2 点 | 已有历史诊断 | RF neck + RF 点数的接口诊断 |

E-2 相对 E-1 同时改变了特征尺度数、projector、HybridEncoder 和采样点布局，所以它估计的是 **neck/interface 整体替换效应**，不是某一个 neck 子层的纯贡献。E-2 与 E-3 才是同一单 P4 接口上的采样点 `6 vs 2`；但两者若不是同批次/同 seed 独立运行，只能先作趋势参考，正式结论应配对重测。

尤其要避免把 E-2 命名为“RF decoder”：RF decoder 是 16 个 cross-attention heads、每 head 2 points；E-2 仍是 EC decoder 的 8 heads 和每 head 6 points。

同批配对的完整结果为：

| 实验 | Params | Best Epoch | F1 | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|
| EC-full neck/interface | 31.365M | 44 | 59.54 | **59.39** | **27.79** | **59.09** | 60.00 |
| strict RF-neck P4 + EC points6 | **27.832M** | 38 | **59.69** | 58.55 | 27.34 | 58.44 | **61.00** |

strict RF-neck 减少 3.533M 参数（约 11.3%），F1 提高 0.15、R 提高 1.00 个百分点，但 mAP50 和 mAP50-95 分别下降 0.84 和 0.45 个百分点。因此 RF 单 P4 接口并未导致巨大退化，它提供了更轻量、F1 相当的替代；但 EC 的 P3/P4/P5 + HybridEncoder 对 COCO AP，尤其更严格 IoU 下的定位质量，仍呈现小幅优势。

由于 E-1 → E-2 同时替换 projector、去掉 HybridEncoder、把三尺度变成单 P4，并改变采样点布局，这 0.45 个百分点只能归因为 **neck/interface 整体效应**，不能单独声称是 HybridEncoder 或某个特征层的贡献。同时差异只有单 seed，下一步更应优先补 seed，而不是立即扩展大量 neck 子层消融。

对应输出目录：

```text
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_neck_pair_ignore9_bs32_2gpu_seed42
outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_points6_ignore9_bs32_2gpu_seed42
```

#### 0.6.3 第二阶段：只做有机制依据的跨块交互

建议优先考虑以下四类交互，不做 `2×2×3×...` 全排列。

##### 交互 1：D-FINE × CDN

CDN 的 DN query 会经过 FDR/FGL/DDF/pre 分支，结构上与 D-FINE 强耦合；因此应补一个 no-D-FINE + no-CDN 角点：

| D-FINE | CDN | 当前状态 |
|---:|---:|---|
| 开 | 开 | 已有 EC-full |
| 开 | 关 | 已有 EC - CDN |
| 关 | 开 | 已有 EC - D-FINE |
| 关 | 关 | **已完成：`EC - D-FINE - CDN`** |

这一 2×2 已完成。结果见 0.2.1：CDN 在 D-FINE-on 下的 mAP50-95 效应为 -0.21，在 D-FINE-off 下为 +0.37，提示 CDN 与 distribution/pre/DDF 路径存在交互，但仍需多 seed 确认。

##### 交互 2：decoder depth × D-FINE，只测端点

不建议把 3/4/5 层与四种 D-FINE 配置全部交叉。当前 3-layer 与 5-layer 端点已完成：

```text
3-layer：EC-full vs EC - D-FINE（已有）
5-layer：EC-full vs `5-layer No D-FINE`（已完成）
```

结果见 0.1.1：D-FINE 在 3 层下 mAP50-95 效应为 -0.33，在 5 层下为 +0.61，初步支持“分布累积和深层自蒸馏需要足够迭代深度”。当前无需再跑 4-layer no-D-FINE，应优先对两个端点补 seed。

##### 交互 3：Mosaic × 最终架构，只做确认

先完成 A×B 的 2×2，再选 mAP50-95 最好的分类/增强组合，分别放到：

```text
最佳 D-FINE 路径
最佳 continuous/no-D-FINE 路径
```

各跑一组确认。不要在 A×B 尚未拆清之前，把 no-Mosaic/Focal 与 decoder 5 层、no-D-FINE、no-CDN 同时叠加，否则即使指标提高也无法解释来源。

##### 交互 4：neck × D-FINE / decoder，只测端点

neck 决定 memory 是三尺度还是单 P4，FDR 和 deformable cross-attention 又直接依赖该 memory，因此 neck 与 D-FINE/decoder 存在明确结构交互。建议分两阶段：

```text
阶段 N1：EC-full neck vs strict RF-neck P4 + points6
         两组都保留 EC decoder、D-FINE、MAL、CDN 与 EC 训练协议。

阶段 N2：仅当 N1 显示显著差异，再在两种 neck 下各跑 strict no-D-FINE；
         形成 neck × D-FINE 的 2×2，判断差异来自特征接口还是 FDR 对单尺度 memory 的适配。
```

不建议立即把 neck 与 decoder 3/4/5、CDN、MAL/Focal、Mosaic 全交叉。若要研究 decoder 联动，先固定 3 层，只比较 EC decoder 与真正 RF decoder 两个端点。

#### 0.6.4 多模块消融与“完全对齐 RF”的建议

多模块实验必须区分 **EC 训练协议下的结构替换** 与 **RF baseline 复现**。推荐建立一条可解释的桥接链：

| 桥接阶段 | Neck | Decoder / query / box | Loss 与训练机制 | 解释边界 |
|---|---|---|---|---|
| R0：EC-full | EC P3/P4/P5 + HybridEncoder | EC decoder + FDR | MAL + GO-LSD + CDN | EC 基线 |
| R1：RF-neck only | strict RF P4 + IdentityEncoder | EC decoder + FDR，先固定 points6 | 其余同 R0 | neck/interface 整体效应 |
| R2：RF-neck + strict no-D-FINE | strict RF P4 | EC continuous decoder | MAL、CDN 保留；无 FDR/GO-LSD/pre | D-FINE 在 RF neck 上的作用；仍不是 RF decoder |
| R3：RF architecture/loss under EC protocol | strict RF P4 | 独立 RF decoder：16-head CA/2 points、learned query、two-stage、共享 head、RF bbox reparameterization | IA-BCE + L1 + GIoU；无 MAL/CDN/D-FINE；仍使用 strict-9、batch32、seed42、EMA/EC schedule | neck 与 decoder/loss 已对齐 RF，但训练协议仍是 EC |
| R4：RF baseline reproduction | strict RF P4 | 与指定 RF 实现完全一致 | 同步 RF 数据口径、seed、batch、优化器、增强、EMA 等协议 | 端到端 RF 复现，不能再与 R0 作单变量归因 |

R3 不应通过给 `ECTransformer` 继续堆 `rf_mode` 开关实现。建议独立注册 `RFTransformerDecoder`、`RFCriterion` 和 RF postprocessor，并用运行时审计确认：P4 `[B,256,40,40]`、3 decoder layers、16 cross-attention heads、2 points、shared class/bbox heads、RF two-stage query、RF box reparameterization及 IA-BCE 全部生效；同时 `pred_corners`、FGL/DDF/LQE/pre、CDN/DN 和 MAL 全部不存在。

因此，“消融到 neck 和 decoder 完全对齐 RF”应以 R3 为终点，而不是把 R1 的 `num_points` 从 6 改成 2。仅有 RF neck + EC decoder points2 仍缺少 RF decoder class、head 数、query 初始化、共享预测头、box parameterization、aux/encoder 输出语义、criterion 和 postprocess，不能命名为完整 RF 对齐。

#### 0.6.5 推荐执行顺序与预算

本轮原计划的五项已全部完成：同批 EC-full、strict RF-neck P4 points6、Mosaic + Focal、5-layer No D-FINE 和 EC - D-FINE - CDN。基于新结果，后续按信息增益建议排序为：

1. **优先补关键端点的 seeds**：`5-layer EC-full vs 5-layer No D-FINE`、`EC-full neck vs strict RF-neck P4 points6`、`Mosaic + MAL vs Mosaic + Focal`。这三组对应当前最关键的深度交互、neck 性价比与分类监督结论。
2. **补 `No Mosaic + MAL`**：仅当需要正式报告 Mosaic 主效应和 Mosaic×loss 交互时运行，用于补齐 A×B 的第四个 strict-9 角点。
3. **暂不补 4-layer No D-FINE**：3/5-layer 端点已经显示交互趋势，增加中间点的信息增益低于多 seed 验证。
4. **neck×D-FINE 暂不扩展**：当前 neck mAP50-95 差异仅 0.45 个百分点。先复核稳定性；只有多 seed 后仍保持明显差异，再补 `RF-neck P4 points6 + no-D-FINE`。
5. **RF 完整对齐保持为独立架构任务**：只有论文需要回答“EC 与 RF 的架构差异来自哪里”时，再实现 R3 独立 RF decoder/loss；不与增强、batch、seed 等 RF 训练协议同时改变。

若每个关键结论需要统计稳定性，应优先对对照与对应消融补 seed，而不是继续增加新的模块组合。建议至少对以下比较跑 3 seeds：

```text
Mosaic + MAL vs No Mosaic + MAL
Mosaic + MAL vs Mosaic + Focal
EC-full vs EC - D-FINE
5-layer EC-full vs 5-layer No D-FINE
EC-full neck vs strict RF-neck P4 + EC points6
```

正式汇报时分别报告每个 seed 的结果及 mean ± std；组合选择必须由验证集决定，测试集只用于最终一次评估。

## 一、CDN 原理与实现

### 1.1 CDN 要解决的问题

EC 的普通 decoder query 来自 encoder memory 的 top-k 选择。训练初期，这些 query 的类别、位置和 reference box 都不准确，Hungarian matching 只能给少数 query 提供直接的 GT 监督，decoder 需要较长时间才能学会把 query 对齐到目标。

CDN（Contrastive Denoising Training）在训练时额外构造带噪声的 GT query，让 decoder 学习：

```text
带噪声的类别和 box
-> 恢复原始类别和 box
```

它属于训练期 query 训练机制，不是 backbone、neck 或 decoder 的一个独立 attention block。推理时不生成 CDN query。

### 1.2 当前代码中的新增参数

ECTransformer 默认配置为：

```yaml
num_denoising: 100
label_noise_ratio: 0.5
box_noise_scale: 1.0
```

当 `num_denoising > 0` 时，decoder 创建一个类别 embedding：

```python
nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
```

strict-9 配置下为 `Embedding(10, 256)`，只有 2560 个参数。它将原始或扰动后的类别 id 转成 DN query content。实现位置为：

```text
ecdetseg/engine/edgecrafter/decoder.py
ecdetseg/engine/edgecrafter/denoising.py
```

### 1.3 DN query 的构造过程

对一个 batch，代码先统计每张图的 GT 数量，并取 batch 内最大值 `max_gt_num`。随后构造 DN group：

1. 将每张图的 GT label 和 box padding 到相同的 `max_gt_num`。
2. 每个 group 复制一套 positive GT 和一套 negative GT。
3. positive query 和 negative query 都加入 box noise。
4. 部分有效 label 被替换为随机类别。
5. label 经过 `denoising_class_embed`，box 经过 `inverse_sigmoid`，形成 DN query content 和 DN reference box。
6. DN query 被放在普通 300 个 query 前面，一起送入同一个 Transformer decoder。

实际 DN query 数量由 group 计算得到：

```text
num_group = max(1, floor(num_denoising / max_gt_num))
actual_dn_queries = 2 * max_gt_num * num_group
```

因此 `num_denoising=100` 是 group 数量的预算，不保证最终 query 数严格等于 100；每组含 positive 和 negative 两套，实际数量通常接近 200。

label noise 的代码概率为 `label_noise_ratio * 0.5`。当前配置下约为 0.25。box noise 对 positive 和 negative 使用不同强度：negative query 的扰动范围更大，用来形成 contrastive denoising。

### 1.4 CDN attention mask

DN query 和普通 query 不能无约束地互相读取，否则普通 query 可能直接看到 GT 构造的重建 query，DN group 也可能互相泄露信息。因此代码构造 `[num_denoising + num_queries, num_denoising + num_queries]` 的 attention mask：

```text
普通 matching query 不能看到 DN query
不同 DN group 之间不能互相看到
```

DN query 仍然可以通过 decoder 的 deformable cross-attention 读取 P3/P4/P5 memory。CDN 不改变 deformable attention 的采样点数量和 reference box 更新方式。

### 1.5 CDN 与普通 decoder 输出

CDN query 与普通 query 使用同一套 decoder layer、分类 head、box head、FDR head 和 LQE。前向结束后，代码依据 `dn_meta['dn_num_split']` 拆分：

```text
dn_outputs       -> DN decoder outputs
normal outputs   -> 普通 final/aux outputs
dn_pre_outputs   -> D-FINE pre 分支的 DN outputs（如果 pre 开启）
```

CDN 不会产生单独的 `loss_cdn`。它是把已有的 detection loss 施加到 DN outputs 上，并加上 `_dn_i` 或 `_dn_pre` 后缀。

### 1.6 CDN 产生的 loss

在 EC-full（FDR/FGL/DDF/GO/CDN 全开）中，每个 DN decoder 层 `i` 可能产生：

```text
loss_mal_dn_i
loss_bbox_dn_i
loss_giou_dn_i
loss_fgl_dn_i
loss_ddf_dn_i
```

如果 D-FINE pre 输出开启，还会产生：

```text
loss_mal_dn_pre
loss_bbox_dn_pre
loss_giou_dn_pre
```

pre 输出没有 `pred_corners`，所以没有 `loss_fgl_dn_pre` 和 `loss_ddf_dn_pre`。

以 3 层 decoder 为例，CDN 的额外 loss 为：

```text
3 × (MAL + L1 + GIoU + FGL + DDF)
+ 1 × (MAL + L1 + GIoU) pre
```

如果是 strict no-D-FINE，distribution、FGL、DDF 和 pre 分支都删除，每个 DN decoder 层只剩：

```text
loss_mal_dn_i
loss_bbox_dn_i
loss_giou_dn_i
```

### 1.7 CDN 的关闭方式

有效 CDN-off 必须设置：

```yaml
ECTransformer:
  num_denoising: 0
```

这样会同时发生：

```text
不创建/不使用 denoising_class_embed
不构造 DN label、DN box 和 DN attention mask
不拼接 DN query
不产生 dn_outputs 或 dn_pre_outputs
criterion 不计算任何 loss_*_dn_* 或 loss_*_dn_pre
```

普通 300 query、MAL、box、FDR/FGL/DDF、GO 和推理输出保持不变。因此“只关闭 DN loss、但保留 DN query”不是有效的 CDN-off 消融。

当前实现中 `decoder.forward()` 调用 CDN 构造函数时将 `box_noise_scale` 传为常数 `1.0`。当前实验配置也是 `1.0`，所以已有结果不受影响；如果以后要比较不同 box noise scale，需要先修正为读取配置字段。

## 二、D-FINE 主模块：FDR 与 GO-LSD

D-FINE 的主要算法模块应按论文术语归纳为：

```text
FDR    = Fine-grained Distribution Refinement
GO-LSD = Global Optimal Localization Self-Distillation
```

当前代码没有名为 `GO-LSD` 的单一类或单一 loss；它由两个开关共同实现：

```text
GO  (Global Optimal matching)       -> use_uni_set
LSD (Localization Self-Distillation)-> use_ddf
```

因此已有配置名 `no_go_ddf` 在算法含义上应阅读为 **GO-LSD-off**：它同时关闭全局最优匹配集合和定位自蒸馏。本文保留该配置文件名，以便和现有任务、输出目录兼容。

### 2.1 组件依赖关系

这几个名字不能简单当成互相独立的三个 loss：

```text
FDR distribution head
  -> 4 条边各 reg_max+1 个 bin
  -> Integral + distance2bbox
  -> FDR box / 下一层 reference

pred_corners
  -> FGL：对 distribution 直接使用 GT corner target
  -> DDF：前层 distribution 向最终层 distribution 蒸馏

GO union
  -> final、decoder aux、pre、encoder 分别执行 Hungarian matching
  -> _get_go_indices 合并各分支的 matching 为跨层 union indices
  -> bbox/GIoU 及启用时的 localization loss 使用该联合监督集合
```

因此：

- GO 可以独立关闭，因为它是匹配集合策略；
- DDF 不能在完全删除 distribution 后保留，因为 DDF 的 student/teacher 都是 distribution；
- FGL 也依赖 `pred_corners`；
- LQE 依赖 distribution quality，因此 distribution 删除时一并关闭。

### 2.2 正式实验链：三组主实验与一组辅助实验

正式消融只保留如下四组，不扩张为“只关 GO”“只关 DDF”等额外排列组合：

| 类别 | 显示名 | 配置 | FDR family | GO | LSD/DDF | 作用 |
|---|---|---|---:|---:|---:|---|
| 主实验 | EC-full | `ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml` | 开 | 开 | 开 | 完整 D-FINE |
| 主实验 | EC - GO-LSD | `ecdet_l_dinov2s_patch16_dec3_liver_no_go_ddf_ignore9.yml` | 开 | 关 | 关 | 保留 FDR，只移除 GO-LSD |
| 主实验 | EC - D-FINE | `ecdet_l_dinov2s_patch16_dec3_liver_no_dfine_ignore9.yml` | 关 | 关 | 不适用 | 连续框基线，两个主模块均删除 |
| 辅助实验 | Continuous + GO | `ecdet_l_dinov2s_patch16_dec3_liver_continuous_with_go_ignore9.yml` | 关 | 开 | 不适用 | 检验 GO 脱离 FDR 后是否仍有效 |

FDR family 在本文中包含 distribution head、Integral/FDR decode、LQE、FGL、`pre_bbox_head` 与 pre auxiliary supervision。GO-LSD 包含 GO union (`use_uni_set`) 和 LSD/DDF (`use_ddf`)。CDN、MAL、deformable cross-attention、backbone、neck、decoder 层数、数据、优化器和训练计划均不属于 D-FINE，四组保持一致。

这不是完整的 2x2 factorial：LSD/DDF 结构性依赖 FDR distribution。FDR 关闭时可以保留 GO，但 DDF 无输入，必须关闭；因此 Continuous + GO 仅作为辅助诊断，不能反向用于计算 FDR 的单独贡献。

### 2.3 主实验：EC - GO-LSD (`no_go_ddf`)

配置文件：

```text
ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_go_ddf_ignore9.yml
```

关键开关：

```yaml
ECTransformer:
  use_fdr_decode: true
  use_aux_distribution: true
  use_lqe: true

ECCriterion:
  use_uni_set: false
  use_fgl: true
  use_ddf: false
```

#### 保留内容

```text
FDR distribution head
Integral / distance2bbox
FDR iterative box refinement
LQE
FGL loss
CDN 及 DN outputs
D-FINE pre head / pre loss
MAL、L1、GIoU
```

#### 删除内容

```text
DDF KL distillation
GO union 在 localization loss 中的使用
```

`use_uni_set=false` 后，`loss_bbox`、`loss_giou` 和 `loss_fgl` 使用各自输出分支的局部 Hungarian matching，不使用跨 final/aux/pre/encoder 的 union indices。当前 criterion 仍会构造 `indices_go`，但关闭后它不再传入这些 loss；这只是冗余索引计算，不会产生模型梯度或改变训练结果。

#### 产生的 loss

普通 decoder、aux、encoder、pre 和 CDN 分支仍会产生原有的 MAL、L1、GIoU；带 distribution 的输出还会产生 FGL。不存在：

```text
loss_ddf
loss_ddf_aux_*
loss_ddf_dn_*
```

#### 能说明的问题

该实验测量：

> 在 FDR/FGL 完全保留的前提下，去掉 DDF 和跨层 GO matching 后，模型性能如何变化。

它可以反映 GO-LSD 对训练监督和中间层定位优化的贡献。

#### 不能说明的问题

它不能单独归因于 GO，也不能说明 FDR 是否重要，因为 FDR 和 FGL 仍然存在。由于 GO 与 LSD 同时关闭，结果也不是一个严格的单因素 GO 消融。

### 2.4 主实验：EC - D-FINE (`no_dfine_ignore9`)

配置文件：

```text
ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_dfine_ignore9.yml
```

关键开关：

```yaml
ECTransformer:
  use_fdr_decode: false
  use_aux_distribution: false
  use_lqe: false
  use_pre_outputs: false

ECCriterion:
  losses: ['mal', 'boxes']
  use_uni_set: false
  use_fgl: false
  use_ddf: false
```

#### 严格 FDR-off 条件

```text
pre_bbox_head 不创建
pre_outputs 不输出
dn_pre_outputs 不输出
所有 *_pre / *_dn_pre loss 消失
```

它仍然保留：

```text
ECTransformer deformable attention
连续 4D iterative box head
MAL
L1 + GIoU
CDN
encoder top-k query 初始化
decoder auxiliary outputs
encoder auxiliary outputs
```

CDN 与 D-FINE 是独立机制，所以 strict no-D-FINE 仍然有 `num_denoising=100`。如果要做 no-D-FINE + no-CDN，必须另外设置 `num_denoising=0`，不能把两者混为一个实验。

#### 产生的 loss

3 层 decoder、CDN 开启时，主要有以下输出组：

```text
final decoder output
2 个普通 decoder auxiliary outputs
1 个 encoder auxiliary output
3 个 DN decoder outputs
```

每个输出组只计算：

```text
loss_mal
loss_bbox
loss_giou
```

因此不会出现任何：

```text
loss_fgl*
loss_ddf*
loss_*_pre
loss_*_dn_pre
```

这是当前实现中真正的 strict no-D-FINE loss 消融，而不是只把最终 FDR box 解码换成连续回归。

#### 能说明的问题

该实验测量：

> 在保留 EC attention、MAL、CDN 和基本 box losses 的前提下，完整移除 D-FINE 的 FDR distribution、FGL、GO-LSD、LQE 和 pre auxiliary path 后，模型性能如何变化。

它是 D-FINE family 的完整算法对照，但仍不是 RF-DETR decoder，也不能直接解释成 RF baseline。

### 2.5 辅助实验：Continuous + GO (`continuous_with_go`)

配置文件：

```text
ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_continuous_with_go_ignore9.yml
```

关键开关：

```yaml
ECTransformer:
  use_fdr_decode: false
  use_aux_distribution: false
  use_lqe: false
  use_pre_outputs: false

ECCriterion:
  losses: ['mal', 'boxes']
  use_uni_set: true
  use_fgl: false
  use_ddf: false
```

它与 EC - D-FINE 使用相同的连续 4D iterative box head，并且同样不创建 distribution head、Integral、LQE、`pre_bbox_head`，也不输出 `pred_corners`、`pre_outputs` 或 `dn_pre_outputs`。唯一算法差异是 `use_uni_set: true`：final、decoder auxiliary、encoder auxiliary 各自先执行 Hungarian matching，随后 `_get_go_indices` 合并为 GO union indices；这些 indices 被 bbox/GIoU loss 使用。GO 不是单独的标量 loss，不改变 MAL 分类 loss，也不作用于 DN loss，DN 继续使用固定的 `get_cdn_matched_indices`。

每个输出组仍只计算：

```text
loss_mal
loss_bbox
loss_giou
```

因此它只用于计算：

```text
Continuous + GO - (EC - D-FINE)
```

即连续回归下 GO matching 的贡献。它不是 FDR 的单独对照；不能使用 `EC-full - (Continuous + GO)` 声称该差异是 FDR 单独贡献。

### 2.6 正式归因方式与历史边界

三组主实验用于计算：

```text
FDR 贡献
= (EC - GO-LSD) - (EC - D-FINE)

GO-LSD 贡献
= EC-full - (EC - GO-LSD)

完整 D-FINE 贡献
= EC-full - (EC - D-FINE)
```

辅助实验只用于：

```text
连续回归下的 GO 贡献
= (Continuous + GO) - (EC - D-FINE)
```

这不是完整 2x2 factorial。原始 DDF 是 LSD 的实现，并以 FDR distribution 为 student/teacher 输入；FDR-off 时可以保留 GO，但 DDF 必须关闭。除非重新定义连续 box distillation，否则不能构造“FDR-off 且 GO-LSD 完整保留”的对照。

`no_fdr_decode` 和 `no_fdr_no_go_ddf` 仅保留为历史诊断实验，不参与主链：前者仍保留 distribution/FGL/DDF 辅助监督，后者默认保留 pre path。因此两者不能替代 EC - D-FINE。

### 2.7 运行时消融审计

正式训练或报告结果前，应从首个 batch 的日志和 runtime output 检查 loss key：

| 配置 | 应存在 | 不应存在 |
|---|---|---|
| EC-full | `loss_fgl*`、`loss_ddf_aux_*`、`loss_ddf_dn_*`、普通 final/aux/encoder/pre/DN/DN-pre 分支对应的 MAL/L1/GIoU | 无 |
| EC - GO-LSD | `loss_fgl*`、普通/pre/DN MAL/L1/GIoU | `loss_ddf*` |
| EC - D-FINE | final/aux/enc/DN 的 MAL/L1/GIoU | `loss_fgl*`、`loss_ddf*`、所有 `*_pre` |
| Continuous + GO | 与 EC - D-FINE 相同的 loss key | `loss_fgl*`、`loss_ddf*`、所有 `*_pre` |

同时检查模型结构：

```text
EC - D-FINE / Continuous + GO 不应创建 dec_bbox_head、Integral、LQE、pre_bbox_head
EC - D-FINE / Continuous + GO 不应输出 pred_corners、pre_outputs、dn_pre_outputs
四组 CDN-on 配置都应存在 DN outputs 和 *_dn_* loss
```

EC-full 的最终普通 decoder layer 是 DDF teacher，因此实际 smoke audit 不应要求普通主分支出现 `loss_ddf`。DDF 主要出现在浅层 decoder auxiliary outputs 的 `loss_ddf_aux_*`，以及相应 DN distribution outputs 的 `loss_ddf_dn_*`。

这样可以区分：

```text
模块未创建
模块创建但只用于辅助监督
模块仍前向但输出未进入 loss
```

只有第一种和第二种在实验归因上是明确的；第三种通常只是无效计算，应避免作为正式消融定义。

## 附录 A：历史诊断配置

以下配置保留用于回溯已有实验和诊断 decoder 路径，但不进入“三组主实验 + 一组辅助实验”的正式归因链。

### A.1 `no_fdr_decode`

配置文件：

```text
ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_decode_ignore9.yml
```

它不是纯粹的 decode-only 单变量消融。结合实际配置和 decoder 代码，它的定义是：

```text
关闭 FDR decode
关闭 LQE
创建连续 4D regression head
保留 distribution head
保留 FGL
保留 DDF
保留 pre_bbox_head 和 pre auxiliary supervision
```

更准确的名称是：

```text
Hybrid 路径诊断：使用连续框作为 decoder reference/output，
同时保留 distribution 辅助预测及 FGL/DDF 监督；LQE 关闭，
pre path 保留。
```

关键配置为：

```yaml
ECTransformer:
  use_fdr_decode: false
  use_aux_distribution: true
  use_lqe: false

ECCriterion:
  use_uni_set: true
  use_fgl: true
  use_ddf: true
```

因此它仍然产生 distribution、FGL、DDF 和 pre 相关 loss；不能表述为完整 FDR-off，也不能用于计算 FDR family 的单独贡献。

### A.2 `no_fdr_no_go_ddf`

配置文件：

```text
ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_no_go_ddf_ignore9.yml
```

它关闭 distribution/FGL/GO-LSD，但由于没有设置 `use_pre_outputs: false`，仍保留 D-FINE pre auxiliary supervision：

```yaml
ECTransformer:
  use_fdr_decode: false
  use_aux_distribution: false
  use_lqe: false

ECCriterion:
  losses: ['mal', 'boxes']
  use_uni_set: false
  use_fgl: false
  use_ddf: false
```

实际仍会创建 `pre_bbox_head`，输出 `pre_outputs`/`dn_pre_outputs`，并产生 `*_pre`/`*_dn_pre` 的 MAL、L1、GIoU loss。因此它不能替代 strict `EC - D-FINE`，只作为历史 continuous-box/no-GO-LSD 诊断。
