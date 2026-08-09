# EdgeCrafter 消融实验说明

本文档记录 EdgeCrafter 检测分支的模型框架、消融边界和后续实验规划。当前文档只定义实验口径，不包含尚未完成的实验结果。模型实现以 `ecdetseg/engine/edgecrafter/` 为准，配置以 `ecdetseg/configs/ecdet/` 为准。

下一位 Codex 接手前先阅读 [`EC_CODEX_HANDOFF.md`](EC_CODEX_HANDOFF.md)，其中记录当前代码改动、架构接口、数据集统计、运行任务和后续检查清单。

## 工作目录

- `ecdetseg/engine/edgecrafter/`：检测模型、backbone、encoder、decoder、criterion 和 CDN 实现。
- `ecdetseg/configs/ecdet/`：ECDet-S/M/L/X 配置。
- `ecdetseg/configs/dataset/`：COCO 和自定义数据集配置。
- `ecdetseg/engine/core/`：YAML 配置解析、组件注册和 workspace 构建。
- `ecdetseg/engine/solver/`：训练、resume、tuning、评估和 checkpoint 保存。
- `ecdetseg/tools/benchmark/`：参数量、FLOPs 和 TensorRT 延迟测量。
- `outputs/`：训练输出目录，由具体配置的 `output_dir` 指定，不同实验不得覆盖。

## EdgeCrafter 检测框架

ECDet 的总体结构为：

```text
image
  -> ViTAdapter / ECViT
  -> multi-scale projection
  -> HybridEncoder
  -> ECTransformer decoder
  -> PostProcessor
```

训练时，`ECCriterion` 对 decoder、auxiliary decoder 和 encoder 输出计算匹配损失；当启用 CDN 时，额外计算 DN query 的损失。

### ECViT backbone

ECViT 不是标准 DINOv2。其主要结构如下：

1. 四层 3x3、stride-2 卷积构成 ConvPyramidPatchEmbed，总 stride 为 16。
2. Transformer 深度为 12，使用 LayerNorm、GELU MLP 和 2D RoPE。
3. 输入 token 包含一个 register token，但不使用标准 DINOv2 的 CLS token 参与空间特征输出。
4. 默认返回第 11、12 个 block 的 token，并对两层取均值。
5. 将 stride-16 token reshape 成空间特征，再通过双线性插值生成 stride 8/16/32 三个尺度。
6. 每个尺度经过 `ConvNormLayer_fuse` 投影到 detector 使用的通道数。

四个官方检测变体的 backbone 宽度如下：

| 模型 | ECViT 变体 | Embed dim | Heads | FFN ratio | projection channels |
| --- | --- | ---: | ---: | ---: | ---: |
| ECDet-S | ECViT-T | 192 | 3 | 4 | 192 |
| ECDet-M | ECViT-T+ | 256 | 4 | 4 | 256 |
| ECDet-L | ECViT-S | 384 | 6 | 4 | 256 |
| ECDet-X | ECViT-S+ | 384 | 6 | 6 | 256 |

当前最适合与标准 DINOv2 ViT-S 对齐的是 ECDet-L，因为两者都使用 384 维、6 个 attention heads、12 层 Transformer。但这只代表张量宽度相同，不代表权重结构兼容。

### Multi-scale projection

ECViT 输出三个特征图后，`HybridEncoder` 接收的默认接口为：

```text
features = [P3, P4, P5]
channels = [256, 256, 256]
strides = [8, 16, 32]
```

这部分 projection 属于 backbone 适配层。后续更换为 RF-DETR 的 `MultiScaleProjector` 时，必须保持三个输出和 stride 契约，不能只输出 RF-DETR 默认的单个 P4。

标准 DINOv2-S/14 在 640 输入下产生约 45x45 patch 网格，而 ECViT 的 patch stride 为 16，对应 40x40 网格。因此 DINOv2 adapter 需要显式将特征重采样到 `80x80、40x40、20x20`，或者同步调整 detector 的 stride、anchor 和评估分辨率。

### HybridEncoder

`HybridEncoder` 是 RT-DETR/D-FINE 派生的多尺度 encoder：

1. 在最粗尺度上执行 AIFI 风格的 self-attention。
2. 通过 top-down FPN 和 bottom-up PAN 进行跨尺度融合。
3. 当前 EdgeCrafter 配置使用 `csp2`、`sum` 融合和不同模型规模的 depth/expansion。

`fuse_op: sum` 在代码中带有 DEIM 来源说明。它必须作为独立结构变量记录，不能默认等价于 MAL 或 Dense O2O。

### ECTransformer decoder

decoder 默认配置为 4 层、300 queries、3 个 feature levels 和 deformable cross-attention。其核心输出包括：

- 分类 logits；
- 连续 box prediction；
- 离散边界分布 `pred_corners`；
- reference points；
- auxiliary decoder outputs；
- CDN 开启时的 DN outputs。

虽然论文使用了 learned object query 的描述，当前代码默认通过 encoder top-k 输出构造 query content；后续实验以代码行为为准。

## Full baseline

本文将原始 EdgeCrafter 检测模型称为 `EC-full`，避免与 RF-DETR 仓库中的 `baseline` 混淆。

`EC-full` 包含：

| 组件 | 当前实现 |
| --- | --- |
| Backbone | 蒸馏后的 ECViT |
| Neck/Projection | ECViT 内置三尺度插值 + 1x1 projection |
| Encoder | HybridEncoder，AIFI + FPN/PAN |
| Decoder | D-FINE 派生 ECTransformer |
| Classification | MAL |
| Localization | L1 + GIoU + FGL + DDF |
| Dense matching | 跨 decoder/encoder 层 union matching |
| Denoising | CDN，`num_denoising=100` |

ECDet-L 可作为第一阶段结构替换的主参考，因为其 384 维 ECViT-S 与标准 DINOv2 ViT-S 的模型宽度一致。若研究目标是 Edge 端效率，也可以同时保留 ECDet-S，但不能将不同模型规模的结果直接混在同一消融表中。

## 模块边界

### DEIM

EdgeCrafter 中的 DEIM 相关机制至少应拆成两个变量：

1. **MAL**：正样本分类目标由匹配框 IoU 构造并取 `gamma` 次幂，负样本权重由 detach 后的预测概率构造。
2. **Dense O2O**：汇总最终 decoder、auxiliary decoder 和 encoder 的 Hungarian matching，形成跨层 union，用于定位相关损失。

因此“去掉 DEIM”有两种口径：

- **算法消融**：只关闭 MAL 和 Dense O2O，保留其余数据增强和 encoder 结构。
- **来源消融**：同时恢复非 DEIM 的分类目标、encoder sum/cat 选择、数据增强和训练策略。

正式结果必须在实验名中注明采用哪一种口径。第一轮推荐使用算法消融，避免一次改变过多变量。

### D-FINE

D-FINE 也要区分两个层次：

- **Loss-only ablation**：关闭 FGL/DDF，保留当前 decoder 的离散边界分布 head。
- **Full decoder ablation**：同时移除离散边界分布、Integral/FDR 相关路径和 D-FINE 专用输出，改回连续 box regression decoder。

Loss-only 结果只能说明 D-FINE 监督损失的贡献，不能表述为“移除了 D-FINE 模块”。完整消融需要单独实现标准连续回归 decoder，并重新检查 pretrained checkpoint 的加载范围。

### CDN

CDN 是训练期的额外 query 分支，不增加推理输出。当前由以下部分组成：

- 正/负 GT noisy queries；
- DN 与 normal query 的 attention mask；
- DN positive index 固定匹配；
- DN final/auxiliary losses。

CDN 消融应使用 `num_denoising=0`，并确认 decoder、criterion、checkpoint loader 和 DDP unused-parameter 行为一致。只删除 DN loss 而保留 DN query 不属于有效的 CDN-off 实验。

当前 `no_cdn` 实现直接继承 DINOv2-S/patch16/3-layer-decoder 的 `ec_full` 配置，只覆盖：

```yaml
ECTransformer:
  num_denoising: 0
```

因此关闭后不会创建 `denoising_class_embed`，训练前向不会构造 DN queries，criterion 也不会产生任何 `loss_*_dn_*`。正常 300 queries、MAL、Dense O2O、FGL/DDF 和推理路径保持不变。

## 第一阶段：原始模块消融

第一阶段先固定 ECViT、EC projection、HybridEncoder 和 decoder，只拆分训练机制。推荐矩阵如下：

| 实验 | MAL | Dense O2O | FGL/DDF | CDN | 目的 |
| --- | :---: | :---: | :---: | :---: | --- |
| `ec_full` | 开 | 开 | 开 | 开 | 原始 EdgeCrafter 参考 |
| `no_dense_o2o` | 开 | 关 | 开 | 开 | Dense O2O 独立贡献 |
| `no_mal` | 关 | 开 | 开 | 开 | MAL 独立贡献 |
| `no_deim` | 关 | 关 | 开 | 开 | 算法层面的 DEIM 移除 |
| `no_dfine_loss` | 开 | 开 | 关 | 开 | FGL/DDF 监督贡献 |
| `no_cdn` | 开 | 开 | 开 | 关 | CDN 训练贡献 |

`no_deim`、`no_dfine_loss` 和 `no_cdn` 应分别与 `ec_full` 比较。不要只用 `no_deim -> no_dfine_loss -> no_cdn` 的串联差值推断独立贡献，因为模块之间存在匹配、query 数量和 loss normalization 交互。

### 当前 DINOv2-S liver 实验

| 实验 | 配置 | 输出目录 | 唯一消融变量 |
| --- | --- | --- | --- |
| `ec_full` | `configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver.yml` | `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_noreg` | CDN 开启，`num_denoising=100` |
| `no_cdn` | `configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_cdn.yml` | `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_noreg_no_cdn` | CDN 关闭，`num_denoising=0` |

两组实验共同使用无 register DINOv2-S、patch16、640x640、3 层 decoder、300 normal queries、总 batch size 32、seed 42、150 epochs、patience 30、AMP、SyncBN、EMA，以及相同 optimizer 和数据增强。两组 decoder 等非 backbone 组件均从随机初始化开始，输出目录和 tmux 会话彼此独立。

2026-08-05 提交记录：

| 实验 | 节点与物理 GPU | tmux 会话 | 状态 |
| --- | --- | --- | --- |
| `ec_full` | `cu04:0,1,2,3` | `ecdet_l_dinov2s_dec3_liver` | 训练中 |
| `no_cdn` | `cu01:4,5,6,7` | `ecdet_l_dinov2s_dec3_liver_no_cdn` | 训练中；首个 step 已确认无 `loss_*_dn_*` |

CDN-off 首次 smoke run 还发现 decoder 的 `pred_segs` 仅在 DN split 分支赋值。当前实现已让非 DN 路径直接使用原始 `pre_segs`；该修复不改变 CDN-on 路径或模型输出定义。

`no_cdn` 四卡运行入口：

```bash
bash scripts/ablation/wait_ecdet_l_dinov2s_patch16_dec3_liver_no_cdn_tmux.sh
```

## 第四阶段：D-FINE / GO-DDF 消融

以下三组实验均继承 `ecdet_l_dinov2s_patch16_dec3_liver.yml`，只改变表中列出的开关；因此不重复复制完整配置。它们固定使用无 register 的 DINOv2-S、patch16、640x640、3 层 decoder、150 epochs、patience 30、seed 42、AMP 和总 batch size 32。

| 实验 | 配置 | 关闭内容 | 输出目录 |
| --- | --- | --- | --- |
| `no_fdr_decode` | `..._no_fdr_decode.yml` | 最终框由连续 4 维回归 head 解码；保留分布辅助、FGL 和 DDF | `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_decode` |
| `no_go_ddf` | `..._no_go_ddf.yml` | 关闭 GO union (`use_uni_set=false`) 和 DDF；保留 FDR/FGL | `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_go_ddf` |
| `no_fdr_no_go_ddf` | `..._no_fdr_no_go_ddf.yml` | 连续框回归；关闭分布辅助、FGL、GO union 和 DDF | `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_no_go_ddf` |

这里的 `no_fdr_decode` 是完整 D-FINE decoder 的 loss-only/路径对照：分布仍作为辅助输出，以保证 FGL/DDF 的比较合法，不能将它表述为“完全移除 FDR”。第三组才是 continuous-box-only decoder。`GO union` 是跨 final、auxiliary、pre 和 encoder Hungarian matches 的 query-GT union；它与 DDF 是两个独立机制，本实验的第二、三组按组合开关同时检验它们的影响。

三个任务由同一个 runner 选择 YAML 和独立输出目录：

```bash
bash scripts/ablation/submit_ecdet_l_dinov2s_patch16_dec3_liver_dfine_ablations.sh
```

提交时按空闲资源固定分配为：`no_fdr_decode -> cu03:0-3`、`no_go_ddf -> cu03:4-7`、`no_fdr_no_go_ddf -> cu04:4-7`；现有 `ec_full` 的 `cu04:0-3` 和 `no_cdn` 的 `cu01:4-7` 不会被覆盖。每个 waiter 以 300 秒（5 分钟）间隔轮询并在目标节点创建独立训练 tmux。`cu03` 没有系统 tmux，仓库 `.local/bin/tmux` 与 `.local/lib/libutempter.so.0` 提供用户态运行时。

## 第五阶段：关闭 Mosaic 增强

`no_mosaic` 继承 D-FINE、FGL、DDF、GO union 和 CDN 全开的 EC 配置，仅将训练 transform 的 `mosaic_prob` 设为 `0.0`；`mosaic_epoch`、MixUp、输入尺寸、优化器和训练计划保持不变。输出目录为 `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_mosaic`。该任务排入 `cu02:4-7`，tmux 会话为 `ecdet_l_dinov2s_no_mosaic`，仍由 5 分钟轮询 waiter 管理。

## 第二阶段：backbone 与 neck 消融

第二阶段在第一阶段确定统一训练口径后，固定 decoder/loss，只改变特征提取和 projection：

| 实验 | Backbone | Projection/neck | 目的 |
| --- | --- | --- | --- |
| `ecvit_ec_projection` | ECViT-S | EC projection | 结构参考 |
| `dinov2_s_ec_projection` | 标准 DINOv2 ViT-S/14 | EC-compatible projection | 只替换 backbone |
| `dinov2_s_rf_projection` | 标准 DINOv2 ViT-S/14 | RF MultiScaleProjector | backbone + neck 替换 |
| `dinov2_s_rf_encoder` | 标准 DINOv2 ViT-S/14 | RF projection + RF-style encoder | 完整 RF 特征路径 |

其中 `dinov2_s_rf_projection` 必须固定输出三层特征。RF-DETR 默认只使用 P4 的配置不能直接接入 EdgeCrafter 的三层 HybridEncoder。

标准 DINOv2-S 实验应加载官方 `facebook/dinov2-small` / `dinov2_vits14` 权重。`ecvits.pth` 是 ECViT 蒸馏权重，`rf-detr-small.pth` 是 RF-DETR 检测 checkpoint，二者不能作为标准 DINOv2-S 的等价初始化。

### EC 使用 RF-DETR neck

`rf_neck` 实验与前述 DINOv2-S EC full 配置保持相同的 D-FINE、FGL、DDF、GO union、CDN、HybridEncoder、3 层 decoder 和训练计划，仅替换 backbone 后的 projection/neck：

- 对齐给定 RF baseline 的 DINOv2-S 阶段输入：RF 配置 `out_feature_indexes=[3,6,9,12]`，在 timm 的零基 block 索引中对应 `[2,5,8,11]`；
- 使用 RF-DETR/LW-DETR `MultiScaleProjector` 的 ConvTranspose/stride-2 sampling、3-block C2f、SiLU 和 channel-wise LayerNorm；
- RF baseline 的 `projector_scale=[P4]` 只输出单尺度，但 EC HybridEncoder 的既有接口要求 P3/P4/P5。为保证本实验只替换 neck 而不同时改 encoder/decoder，RF projector 在 EC 中配置为 `P3/P4/P5`，输出分别为 `256x80x80、256x40x40、256x20x20`；
- DINOv2-S backbone 继续加载同一无 register checkpoint；RF neck、HybridEncoder、decoder 和 heads 随机初始化。

因此本实验应命名为 `EC + RF multi-scale neck`，不能表述为完整复制 RF baseline 的单 P4 backbone 接口。配置为 `ecdet_l_dinov2s_patch16_dec3_liver_rf_neck.yml`，输出目录为 `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck`，tmux 会话为 `ecdet_l_dinov2s_rf_neck`。

### RF P4-only 严格对齐

前述 multi-scale 版本保留了 EC HybridEncoder，参数增加且不等价于 RF baseline。严格对齐版本使用以下路径：

```text
DINOv2-S block 3/6/9/12
  -> RF MultiScaleProjector(scale=P4)
  -> 256x40x40, stride 16
  -> IdentityEncoder
  -> single-level EC D-FINE decoder
```

下面明确区分前述 EC-compatible 多尺度 neck 与严格 RF P4 neck；两者都使用 EC 的 D-FINE decoder，但不是同一个特征接口：

| 项目 | EC-compatible RF multi-scale neck（`rf_neck`） | 严格 RF P4 neck（`rf_neck_p4_ignore9`） |
| --- | --- | --- |
| DINOv2 中间 block | 3/6/9/12 | 3/6/9/12 |
| RF projector 输出 | P3/P4/P5 | 仅 P4 |
| projector scale | `[2.0, 1.0, 0.5]` | `[1.0]` |
| 输出特征 | `[256x80x80, 256x40x40, 256x20x20]` | `[256x40x40]` |
| EC HybridEncoder | 保留 | 移除，使用 `IdentityEncoder` |
| EC decoder 输入 | 3 层 | 1 层 |
| `num_levels` / `num_points` | `3` / `[3, 6, 3]` | `1` / `[2]` |
| 类别与数据 | 13 类训练，评估时过滤 9-12 | train/valid/test 均为严格 9 类 JSON |
| 参数量 | 41.70M | 27.76M |
| 实验含义 | 将 RF projector 插入 EC 多尺度流水线 | RF P4 neck 直接连接 EC D-FINE decoder |

严格 P4 任务已提交到 `cu02:0-3`，本地 waiter 为 `wait_ec_rf_neck_p4_ignore9`，训练 tmux 会话为 `ecdet_l_dinov2s_rf_neck_p4_ignore9`。首次运行在 epoch 0 评估后因 `ec_solver.py` 对标量 `test_stats` 进行迭代而退出；该 solver bug 已修复，当前 waiter 正按 300 秒轮询 GPU 空闲状态，训练启动后再生成正式指标，不能将现有 smoke 结果填入完成结果表。

配置为 `ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_ignore9.yml`。关键变量为：`interaction_indexes=[2,5,8,11]`、`rf_scale_factors=[1.0]`、`num_levels=1`、`feat_strides=[16]`、`num_points=[2]`，decoder 仍为 3 层，FDR/FGL/DDF/GO union/CDN 均开启。该模型参数量为 27.76M（27,757,881），输出 head 为 9 类。

为对齐 RF 数据口径，这一版本不是只在评估时 ignore，而是训练、验证和测试 JSON 都删除 category ID `9-12`，并设置 `num_classes=9`。派生标注由 `scripts/ablation/make_liver_ignore_9_12_annotations.py` 生成：

- `train_ignore_9_12.json`：29,540 张图，30,630 个标注；
- `valid_ignore_9_12.json`：2,504 张图，2,650 个标注；
- `test_ignore_9_12.json`：3,315 张图，3,973 个标注。

输出目录为 `outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_ignore9`，四卡 tmux 为 `ecdet_l_dinov2s_rf_neck_p4_ignore9`。由于它采用 9 类训练，正式归因 RF P4 neck 的贡献时还需要一个同样 9 类训练的 EC-full 对照；不能直接把它与 13 类训练、评估时才 ignore 的旧 EC-full 当作完全严格的单变量比较。

## 第三阶段：完整结构替换

只有第二阶段确定了特征接口后，才考虑替换 encoder 或 decoder：

1. ECViT + EC projection + HybridEncoder + ECTransformer。
2. DINOv2-S + RF projection + HybridEncoder + ECTransformer。
3. DINOv2-S + RF projection + RF-style encoder + ECTransformer。
4. DINOv2-S + RF projection + RF-style encoder + RF decoder。

每一步只增加一个结构变化。RF decoder 的 query、feature level、position encoding 和 box head 与 ECTransformer 不同，不能把它归为普通 neck 消融。

## 公共训练设置

正式实验必须固定以下条件：

- 数据集、train/val 划分和类别顺序；
- 输入分辨率，默认 640x640；
- batch size、梯度累计和 worker 数；
- optimizer、学习率、warmup、epoch 和增强关闭时机；
- seed、AMP、SyncBN、EMA 和梯度裁剪；
- evaluator、`num_top_queries` 和 checkpoint 选择规则。

改变 backbone 后，如果预训练来源从 ECViT 蒸馏权重切换为官方 DINOv2 权重，应把它记录为独立的初始化变量，不能仅写成“backbone 替换”。

## 评估和 checkpoint

每个实验使用独立的 `output_dir`，至少保存：

- `last.pth`：最近一次训练状态；
- `best.pth`：按统一验证指标保存的最佳模型；
- `log.txt`：训练和验证日志；
- `eval/`：COCO evaluator 输出；
- 配置文件副本和源码版本信息。

结果表统一报告同一 best epoch 的 AP、AP50、AP75、APS、APM、APL、参数量、GFLOPs 和延迟。延迟必须注明设备、batch、精度和部署后端。只有一个 seed 时，小于约 0.5 AP 的差异只能视为待复现信号。

## 统一 Ignore 评估范式

数据集 JSON 中的 category ID `9,10,11,12` 对应名称 `16,17,18,19`，与 RF-DETR baseline 的 9 类口径不一致。后续 EC 结果默认采用以下评估范式：

- 训练 checkpoint 不变，不重新训练；
- evaluator 过滤 GT 和预测中的 category ID `9,10,11,12`，仅评估 category ID `0-8`；
- 使用每个实验训练期间按原始验证指标选出的 `best.pth`；
- 本次汇总在 `test.json` 上重评；Best epoch 仍是原始 13 类验证集选出的 epoch，因此它不等价于重新训练 9 类 head；
- `mAP@50` 和 `mAP@50:95` 使用 COCO bbox evaluator；F1 定义为 IoU=0.50 COCO precision-recall 曲线上的最大 macro-F1，并非固定置信度阈值下的 F1。

统一入口为 `scripts/ablation/evaluate_ecdet_liver_ignore_9_12.sh`，通过 `evaluator.ignore_category_ids=[9,10,11,12]` 开启过滤；不传该参数时保持原始 13 类行为。

## 分组实验结果分析

共同设置：无 register DINOv2-S、patch16、640x640、3 层 decoder、150 epochs、patience 30、seed 42、AMP、四卡训练。除严格 RF P4 外，下面已完成结果均是 **13 类训练 checkpoint 的 evaluator-only ignore-9-12 重评**，指标来自同一个 best epoch 的 test split。

### 第一组：EC-full 基线

配置：`ecdet_l_dinov2s_patch16_dec3_liver.yml`
输出：`outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_noreg`

结构为 DINOv2-S 无 register backbone、EC projection、HybridEncoder 和 3 层 EC D-FINE decoder。CDN、FDR、FGL、DDF、GO union 和 Mosaic 均开启。DINOv2-S 是唯一加载预训练的部分，projection、HybridEncoder、decoder、head 和 criterion 随机初始化。

结果：`31.37M`，best epoch `45`，F1 `0.595`，mAP@50 `0.600`，mAP@50:95 `0.286`。

这是当前所有训练机制、特征接口和评估结果的参考点。后续实验只能在明确写出唯一变化后与该结果比较，不能把不同类别训练口径或不同 decoder 层数的模型混入该基线。

### 第二组：CDN 消融

配置：`ecdet_l_dinov2s_patch16_dec3_liver_no_cdn.yml`
唯一结构变量：`ECTransformer.num_denoising: 100 -> 0`。

关闭后不构造 DN queries、DN attention mask、DN class embedding 和 DN losses；normal query、MAL、Dense O2O、FGL/DDF、HybridEncoder 和推理输出保持不变。

结果：`31.36M`，best epoch `44`，F1 `0.595`，mAP@50 `0.594`，mAP@50:95 `0.274`。

相对 EC-full，mAP@50:95 下降 `0.012`，是本轮最明显的训练机制变化。F1 没有变化，说明 F1 对 CDN 的差异不如 AP 指标敏感。该结论仍是单 seed 信号，不能直接宣称 CDN 在所有数据集上都带来同幅度收益。

### 第三组：FDR decode / GO-DDF 消融

这一组包含三个相关但不等价的实验：

| 实验 | 关闭内容 | 保留内容 | 结果（mAP@50:95） |
| --- | --- | --- | ---: |
| `EC - FDR decode` | 最终框不使用分布 FDR 解码，改用连续 4 维回归解码 | 分布辅助输出、FGL、DDF、GO union | 0.282 |
| `EC - GO-DDF` | `use_uni_set=false`，同时关闭 DDF | FDR/FGL | 0.283 |
| `EC - FDR - GO-DDF` | 连续框回归、关闭分布辅助、FGL、GO union 和 DDF | 其余 decoder/训练设置 | 0.284 |

对应结果为：

- `EC - FDR decode`：31.76M，best epoch 38，F1 0.589，mAP@50 0.596，mAP@50:95 0.282；
- `EC - GO-DDF`：31.37M，best epoch 37，F1 0.592，mAP@50 0.599，mAP@50:95 0.283；
- `EC - FDR - GO-DDF`：31.27M，best epoch 45，F1 0.596，mAP@50 0.600，mAP@50:95 0.284。

解释边界：`no_fdr_decode` 是 D-FINE decoder 的路径/loss-only 对照，不是完全移除 D-FINE；分布辅助仍存在以便继续计算 FGL/DDF。第三个联合实验才接近 continuous-box-only，但仍保留 ECTransformer、MAL 和其他 decoder 结构。单项关闭约损失 `0.003-0.004` mAP，联合关闭没有呈现简单加和，说明 FDR、GO union 和 DDF 之间存在交互。

### 第四组：Mosaic 消融

配置：`ecdet_l_dinov2s_patch16_dec3_liver_no_mosaic.yml`
唯一变量：训练 transform 的 `mosaic_prob=0.0`。

`mosaic_epoch`、MixUp、输入尺寸、优化器、decoder、CDN、FDR/FGL/DDF、GO union 和训练计划保持不变。结果为 `31.37M`，best epoch `40`，F1 `0.591`，mAP@50 `0.593`，mAP@50:95 `0.286`。

与 EC-full 相比，mAP@50:95 相同，mAP@50 下降 `0.007`。当前只能说明单 seed 下 Mosaic 的收益不明显，不能据此认定 Mosaic 无效；应通过多 seed 或更稳定的验证曲线确认。

### 第五组：EC-compatible RF multi-scale neck

配置：`ecdet_l_dinov2s_patch16_dec3_liver_rf_neck.yml`
输出：`outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck`

路径为：

```text
DINOv2 block 3/6/9/12
  -> RFMultiScaleProjector(scale=[2.0,1.0,0.5])
  -> P3/P4/P5: [256x80x80,256x40x40,256x20x20]
  -> EC HybridEncoder
  -> 3-level EC D-FINE decoder
```

该实验保留 EC HybridEncoder 和三层 decoder，只把 backbone 后的 projection 换成 RF 风格。RF projector 通过四个中间 block 的拼接、转置卷积/stride-2 sampling、C2f bottleneck、LayerNorm 和 SiLU 生成三层特征。

结果：`41.70M`，best epoch `37`，F1 `0.592`，mAP@50 `0.585`，mAP@50:95 `0.279`。

参数量比 EC-full 增加约 `10.33M`，原因是三套 RF C2f 和采样卷积叠加在原有 HybridEncoder 之前，并不是 DINOv2 backbone 参数增加。该结果低于 EC-full，说明当前单 seed 和 EC HybridEncoder 接口下，RF projector 的额外容量没有转化成收益；不能把这个结果当作严格 RF P4 neck 的结论。

### 第六组：严格 RF P4 对齐任务

配置：`ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_ignore9.yml`
输出：`outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_rf_neck_p4_ignore9`
本地 waiter：`wait_ec_rf_neck_p4_ignore9`
目标训练资源：`cu02:0-3`
训练 tmux：`ecdet_l_dinov2s_rf_neck_p4_ignore9`

严格路径为：

```text
DINOv2 block 3/6/9/12
  -> RFMultiScaleProjector(scale=[1.0])
  -> [256x40x40], stride=16
  -> IdentityEncoder
  -> single-level EC D-FINE decoder
```

关键配置：`num_classes=9`、`num_levels=1`、`feat_strides=[16]`、`num_points=[2]`、`num_layers=3`。它严格对齐 RF 的单 P4 neck 接口，但 decoder 仍是 EC D-FINE decoder，不是完整 RF-DETR decoder。

该任务使用真正删除 category ID `9-12` 的 9 类 train/valid/test JSON，而不是训练 13 类后仅在 evaluator 过滤。当前参数量为 `27.76M`。

首次运行在 epoch 0 评估后因 `ec_solver.py` 将 F1 标量当作可迭代对象而退出；solver 已修复。当前 waiter 仍在运行并每 300 秒检查 `cu02:0-3`，最近轮询显示四张卡仍未达到空闲显存阈值，远端训练 tmux 尚未创建。因此该严格对齐任务仍在运行队列中，正式指标暂时不能填写，也不能把 epoch 0 smoke 结果当成最终结果。

严格 RF P4 的正式归因还需要同样 9 类训练的 EC-full 对照；不能直接把它与旧的 13 类训练、evaluator-only ignore 的 EC-full 做严格单变量比较。

## 汇总大表

下表统一列出当前各组实验。除最后一行外，所有指标都是 13 类训练 checkpoint 的 ignore-9-12 test 重评；最后一行是真正 9 类训练任务，当前仍在等待 GPU 空闲并继续运行。

| 实验组 | Model | 唯一变化/neck | 数据口径 | Params | Best epoch | F1 | mAP@50 | mAP@50:95 | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| EC 基线 | EC-full | EC projection + HybridEncoder | 13 类训练，评估过滤 9-12 | 31.37M | 45 | 0.595 | 0.600 | 0.286 | 已完成 |
| CDN | EC - CDN | `num_denoising=0` | 13 类训练，评估过滤 9-12 | 31.36M | 44 | 0.595 | 0.594 | 0.274 | 已完成 |
| FDR/GO-DDF | EC - FDR decode | 连续最终框解码，保留分布辅助/FGL/DDF | 13 类训练，评估过滤 9-12 | 31.76M | 38 | 0.589 | 0.596 | 0.282 | 已完成 |
| FDR/GO-DDF | EC - GO-DDF | 关闭 GO union 和 DDF | 13 类训练，评估过滤 9-12 | 31.37M | 37 | 0.592 | 0.599 | 0.283 | 已完成 |
| FDR/GO-DDF | EC - FDR - GO-DDF | 连续框、关闭分布辅助/FGL/GO/DDF | 13 类训练，评估过滤 9-12 | 31.27M | 45 | 0.596 | 0.600 | 0.284 | 已完成 |
| Mosaic | EC - Mosaic | `mosaic_prob=0` | 13 类训练，评估过滤 9-12 | 31.37M | 40 | 0.591 | 0.593 | 0.286 | 已完成 |
| Neck | EC + RF multi-scale neck | RF P3/P4/P5 projector，保留 HybridEncoder | 13 类训练，评估过滤 9-12 | 41.70M | 37 | 0.592 | 0.585 | 0.279 | 已完成 |
| Neck | EC + strict RF P4 neck | RF 单 P4 projector，IdentityEncoder，1-level decoder | 严格 9 类训练 | 27.76M | pending | - | - | - | **仍在跑/等待 GPU** |

总表阅读时要注意：参数量和指标不能脱离数据类别口径解释；RF multi-scale 与 strict RF P4 也不能混称。strict RF P4 的结果只有在任务完成并与严格 9 类 EC-full 对照后，才适合用于正式 neck 归因。

## 实现约定

- 每个新变量优先增加配置开关，不改变 `ec_full` 默认行为。
- 每个实验至少配套一个 YAML、一个 forward/shape 测试和一个配置回归测试。
- backbone 输出必须显式记录 channels、spatial size 和 stride。
- 预训练权重加载必须输出 matched、missing 和 unexpected keys；禁止静默 `strict=False` 后不记录缺失项。
- 关闭模块时同时检查训练、验证、resume、tuning、ONNX 导出和 DDP unused parameters。
- 实验结果不得覆盖已有输出目录；实验名必须编码结构变量和初始化来源。
- 正式训练前先运行单 batch forward、loss backward、1 epoch smoke 和 FLOPs/参数量检查。

## 运行入口规划

当前 EdgeCrafter 使用：

```bash
cd ecdetseg
python train.py -c configs/ecdet/ecdet_l.yml
```

后续推荐新增统一的实验配置目录和脚本入口，例如：

```text
ecdetseg/configs/ablation/
ecdetseg/scripts/ablation/
ecdetseg/outputs/ablation/
```

每个实验先支持配置解析和 1 epoch smoke，再提交正式训练。本文档中的实验名和变量定义先作为接口约定，待第一组实验实现后同步补充具体 YAML 路径和结果。
