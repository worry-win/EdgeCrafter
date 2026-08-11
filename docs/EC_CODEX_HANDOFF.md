# EdgeCrafter Codex 接手文档

> 更新时间：2026-08-09
> 工作目录：`/cobot/Code/wanrui/EdgeCrafter`

这份文档用于下一位 Codex 接手当前 EdgeCrafter 检测消融工作。接手后必须先阅读本文和 [`EC_ABLATIONS.md`](EC_ABLATIONS.md)，再修改训练代码、配置或启动任务。当前研究重点是：标准无 register DINOv2-S backbone、EC/D-FINE decoder、RF-DETR 风格 neck，以及 liver 数据集上的统一消融。

## 1. 当前状态

| 项目 | 当前状态 |
| --- | --- |
| 主要数据集 | `/cobot/Data/Lesion_det/det_liver` |
| DINOv2-S dec3 基线 | `ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_ec_full_ignore9.yml` |
| DINOv2-S dec5 对照 | `ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec5_liver_ignore9.yml`；只改变 decoder depth |
| Backbone | timm `vit_small_patch14_dinov2`，运行时 patch size 改为 16，12 blocks，384 dim |
| 预训练部分 | 仅 DINOv2-S backbone 加载权重；projection、encoder、decoder、heads、criterion、EMA 随机初始化 |
| Decoder 对照 | dec3 与 dec5；每层均使用三尺度 MSDeformableAttention 和 `[3,6,3]` 采样点 |
| 训练计划 | 150 epochs，patience 30，seed 42，AMP，四卡，总 batch 32 |
| Ignore 口径 | 当前正式对照均使用严格 9 类 JSON；旧 13 类 checkpoint 的 evaluator-only ignore 只能作为历史结果 |
| 已完成 ECDet-X 对照 | Job `795_0/1`；双卡、每卡 16、50 epochs，均已训练并自动完成 strict-9 test |
| 当前 ECDet-X 任务 | Job `843_0/1`；四卡、每卡 8、150 epochs、patience 30，分别加载 COCO 与 moreOrgan 完整检测器权重 |
| 当前 DINOv2-S dec5 | Job `842`；四卡、每卡 8、150 epochs、patience 30，仅加载 no-register backbone，5 层 decoder 随机初始化 |
| 严格 RF P4 | 已完成训练与 test；Best Epoch 54，mAP50-95 0.274 |
| strict no-D-FINE | 已实现、未提交；删除 FDR/LQE/FGL/DDF/GO 与 D-FINE pre head/pre loss，保留 CDN 与 MAL |
| Git | 用户已重新启用 Git 操作；仍不得执行 `reset --hard`、`clean` 或批量还原工作树 |

当前优先监控 `842/843`，不要重复提交同配置、同输出目录任务：

```bash
squeue -j 842,843 -o '%.12i %.24j %.10P %.10T %.12M %.4D %R'
tail -f outputs/slurm/ecdino-dec5-4g150-842.out
tail -f outputs/slurm/ecx-liver9-4g150-843_0.out
tail -f outputs/slurm/ecx-liver9-4g150-843_1.out
```

## 2. 必须先阅读的代码

按以下顺序阅读，避免只看 YAML 就修改结构：

| 优先级 | 文件/目录 | 接手时必须理解的内容 |
| ---: | --- | --- |
| 1 | `docs/EC_ABLATIONS.md` | 实验命名、ignore 范式、已完成结果、RF-compatible 与 strict RF P4 的边界 |
| 2 | `docs/EC_DINOV2_DECODER_3_VS_5.md` | DINOv2-S dec3/dec5 的配置继承、预训练边界、逐层结构和严格差异 |
| 3 | `ecdetseg/configs/ecdet/ecdet.yml` | 官方 EC 默认结构：HybridEncoder、4 层默认 decoder、300 queries、CDN、FDR/FGL/DDF、GO union |
| 4 | `ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver.yml` | liver 基线的显式覆盖：DINOv2-S、patch16、3 层 decoder、训练计划和增强 |
| 5 | `ecdetseg/engine/edgecrafter/dinov2_adapter.py` | DINOv2 权重加载、patch14 到 patch16 重采样、中间 block 提取、EC projection 和 RF projector |
| 6 | `ecdetseg/engine/edgecrafter/modeling.py` | `ECDet` 的 backbone -> encoder -> decoder 调用链，以及 strict P4 的 `IdentityEncoder` |
| 7 | `ecdetseg/engine/edgecrafter/hybrid_encoder.py` | AIFI 风格 self-attention、top-down FPN、bottom-up PAN、多尺度融合和 stride 契约 |
| 8 | `ecdetseg/engine/edgecrafter/decoder.py` | ECTransformer、deformable cross-attention、query 选择、FDR、LQE、auxiliary outputs、CDN 分支 |
| 9 | `ecdetseg/engine/edgecrafter/criterion.py` | Hungarian matching、MAL、bbox/GIoU、FGL、DDF、Dense O2O union、DN loss |
| 10 | `ecdetseg/engine/edgecrafter/denoising.py` | CDN query 构造、正负 noisy query、attention mask、DN positive index |
| 11 | `ecdetseg/engine/solver/ec_engine.py` | 训练/验证循环、COCO evaluator、F1 计算、`test_stats` 的返回结构 |
| 12 | `ecdetseg/engine/solver/ec_solver.py` | epoch 训练、best checkpoint、early stopping、resume；已修复标量 F1 导致的错误 |
| 13 | `ecdetseg/engine/data/dataset/coco_eval.py` | evaluator 的 `ignore_category_ids` 过滤实现，确认它只影响评估，不会改变训练 head |
| 14 | `scripts/ablation/run_ecdet_l_dinov2s_patch16_dec3_liver_dfine_ablation.sh` | 各实验 YAML 选择、输出目录、resume、strict P4 的 test annotation 切换 |
| 15 | `scripts/ablation/submit_ecdet_l_dinov2s_patch16_dec3_liver_dfine_ablations.sh` | waiter 分配节点/GPU、5 分钟轮询和严格 RF P4 的固定资源 |
| 16 | `scripts/ablation/wait_ecdet_l_dinov2s_patch16_dec3_liver_tmux.sh` | 远端节点检测、空闲显存阈值、远端 tmux 启动规则 |
| 17 | `scripts/ablation/evaluate_ecdet_liver_ignore_9_12.sh` | 13 类 checkpoint 的 evaluator-only ignore 重评入口 |
| 18 | `scripts/ablation/make_liver_ignore_9_12_annotations.py` | 删除 category ID `9-12` 并生成真正 9 类 train/valid/test JSON |

还必须对照 RF-DETR 源码，不要凭文件名假设 neck 已经完全复制：

| RF-DETR 文件 | 对照重点 |
| --- | --- |
| `/cobot/Code/wanrui/rf-detr-wr/src/rfdetr/models/backbone/projector.py` | 官方 `LayerNorm`、`ConvX`、`Bottleneck`、`C2f`、`MultiScaleProjector` |
| `/cobot/Code/wanrui/rf-detr-wr/src/rfdetr/models/backbone/backbone.py` | `out_feature_indexes`、`projector_scale`、P3/P4/P5/P6 与 encoder 输入接口 |
| `/cobot/Code/wanrui/rf-detr-wr/docs/WR_CODEX_HANDOFF.md` | RF-DETR 当前实验口径、D-FINE、Denoise、DEIM 术语和结果记录规范 |
| `/cobot/Code/wanrui/rf-detr-wr/docs/WR_ABLATIONS.md` | RF 侧消融边界，特别是 neck、decoder 和 loss-only 消融的区别 |

原始论文、EC 代码说明和本仓库实现要分开核对：论文中的模块命名不能替代当前 forward 代码行为。若论文、RF-DETR 实现和 EdgeCrafter 当前代码不一致，实验记录必须以实际配置和 forward 为准，并把差异写入文档。

## 3. 当前模型架构

### 3.1 EC/DINOv2-S 三尺度路径

当前 liver 基线的实际调用链是：

```text
640x640 image
  -> timm DINOv2-S ViT
     12 transformer blocks, embed_dim=384, 6 heads
     patch14 checkpoint resized to runtime patch16 / 40x40 grid
  -> get_intermediate_layers(indexes=[10,11])
  -> two final features mean-pool in feature dimension
  -> bilinear resize: 2.0x / 1.0x / 0.5x
  -> three random 1x1 Conv + BatchNorm projections
  -> [B,256,80,80], [B,256,40,40], [B,256,20,20]
  -> EC HybridEncoder, strides [8,16,32]
  -> ECTransformer, 3 decoder layers in current `dec3` configs
  -> PostProcessor
```

注意：DINOv2 仍然是 12 blocks；`interaction_indexes=[10,11]` 只表示默认基线抽取最后两个 block，不表示 backbone 只有两个 block。ECViT 官方实现也是 12 blocks，但其 patch embedding 是四层 stride-2 卷积，和 ViT 的一维 patch projection 不同。

### 3.2 严格 RF P4 neck 路径

严格 RF P4 任务的实际路径是：

```text
DINOv2-S blocks 3/6/9/12
  -> timm zero-based indexes [2,5,8,11]
  -> RFMultiScaleProjector(scale_factors=[1.0])
  -> RFC2f / RF-style channel LayerNorm / SiLU
  -> [B,256,40,40], stride=16
  -> IdentityEncoder
  -> single-level ECTransformer
     feat_channels=[256], feat_strides=[16]
     num_levels=1, num_points=[2], num_layers=3
  -> EC D-FINE decoder outputs
```

这里的“严格 RF-neck”只严格对齐 neck 的单 P4 特征接口，不是完整 RF-DETR decoder。decoder 仍然是 EdgeCrafter 的 ECTransformer，仍使用 EC 侧的 FDR/FGL/DDF/GO union/CDN 配置。

### 3.3 两个 RF neck 实验不能混称

| 项目 | `rf_neck` 兼容版 | `rf_neck_p4_ignore9` 严格版 |
| --- | --- | --- |
| 中间特征 | block 3/6/9/12 | block 3/6/9/12 |
| projector 输出 | P3/P4/P5 | 仅 P4 |
| scale factors | `[2.0,1.0,0.5]` | `[1.0]` |
| tensor shape | 256x80x80、256x40x40、256x20x20 | 256x40x40 |
| EC HybridEncoder | 保留 | 删除，换 `IdentityEncoder` |
| decoder levels | 3 | 1 |
| points | `[3,6,3]` | `[2]` |
| 数据类别 | 13 类训练，评估过滤 9-12 | 从训练开始即 9 类 |
| 参数量 | 41.70M | 27.76M |
| 结论范围 | RF projector 插入 EC 多尺度流水线 | 单尺度 RF P4 neck 接 EC decoder |

兼容版的 41.70M 结果不能当作严格 RF P4 的结果。严格版移除 HybridEncoder 后参数量下降，也改变了 decoder 的 feature level 和 deformable sampling 结构。

### 3.4 Decoder 层数的历史差异

- `ecdet.yml` 官方默认值是 `ECTransformer.num_layers=4`。
- 原 liver 消融配置文件名含 `dec3`，显式覆盖为 `num_layers=3`。
- 当前已新增严格 DINOv2-S dec5 对照：`ecdet_l_dinov2s_patch16_dec5_liver_ignore9.yml`，Job `842`。
- 解析完整 YAML 后，dec5 相对 strict-9 dec3 的模型和训练配置只有 `ECTransformer.num_layers: 3 -> 5`；输出目录和 Slurm 资源请求不属于算法差异。
- “增加两层”会自动增加两套完整 `TransformerDecoderLayer + score head + 132-dim FDR head + LQE`，并增加两组 auxiliary outputs。
- dec3 和 dec5 的每一层都使用 `MSDeformableAttention`、3 个 feature levels 和 `[3,6,3]` 采样点；不是从普通 cross-attention 改为 deformable attention。
- dec3/dec5 都只加载同一 DINOv2-S no-register backbone；projection、HybridEncoder、decoder 和 heads 均随机初始化。

完整说明见 [`EC_DINOV2_DECODER_3_VS_5.md`](EC_DINOV2_DECODER_3_VS_5.md)。

### 3.5 ECDet-X 完整权重微调路径

另有一组与 DINOv2-S backbone-only 基线不同的 ECDet-X 实验：

| 初始化 | checkpoint | 原类别数 | 加载范围 |
| --- | --- | ---: | --- |
| COCO decoder | `/cobot/Code/CODE/EC/ckpts/EC-1+2+coco-decoder.pth` | 80 | 完整 backbone、neck、4-layer decoder 和回归头 |
| moreOrgan | `/cobot/Code/CODE/EC/output/stage2_finetune_moreOrgan_freeze_encoder/best.pth` | 71 | 完整 backbone、neck、4-layer decoder 和回归头 |

两个 checkpoint 与当前 ECDet-X 精确匹配 764 个张量。因为目标任务是 9 类，以下部分重新初始化：

- `decoder.denoising_class_embed.weight`；
- encoder score head；
- 4 个 decoder score heads。

moreOrgan checkpoint 额外含 9 个 `_feataug_crop_head.*` 张量。当前仓库默认 ECDet-X 没有 FeatAug 模块，这些额外张量会被忽略；因此当前实验不是 FeatAug 实验。

## 4. 已修改内容

以下是当前工作树中与本轮实验直接相关的修改摘要。没有 Git commit 可依赖，接手时应以文件现状为准。

### 4.1 Backbone adapter

文件：`ecdetseg/engine/edgecrafter/dinov2_adapter.py`

- 新增 `RFChannelLayerNorm`：对 NCHW 特征转为 NHWC 后按 channel 做 LayerNorm。
- 新增 `RFConvX`、`RFBottleneck`、`RFC2f`。
- 新增 `RFMultiScaleProjector`，支持 scale `2.0/1.0/0.5`，并可通过 `rf_scale_factors=[1.0]` 只输出 P4。
- `DinoV2Adapter` 增加 `projector_type='ec'|'rf'`、`rf_num_blocks`、`rf_scale_factors` 开关。
- EC projection 保持三尺度接口；RF projection 使用多 block feature 融合。
- 权重加载会检查 checkpoint、过滤 `mask_token`，并对 patch projection 与 `pos_embed` 做尺寸重采样。
- 加载必须打印 matched/missing/unexpected 结果；不能静默忽略不匹配。

### 4.2 Encoder/model registration

文件：`ecdetseg/engine/edgecrafter/modeling.py`、`ecdetseg/engine/edgecrafter/__init__.py`

- 注册 `IdentityEncoder`，用于严格 RF P4 neck 直接把 `[B,256,40,40]` 传给 decoder。
- `ECDet.forward_features()` 仍然固定执行 backbone -> encoder；因此 strict P4 必须通过注册的 IdentityEncoder 接入，不能把 encoder 字符串随意改成 `None`。

### 4.3 配置与脚本

- `ecdet_l_dinov2s_patch16_dec3_liver.yml`：当前 DINOv2-S/patch16/dec3 liver 基线。
- `..._no_cdn.yml`、`..._no_fdr_decode.yml`、`..._no_go_ddf.yml`、`..._no_fdr_no_go_ddf.yml`、`..._no_mosaic.yml`：第一阶段训练机制消融。
- `..._no_dfine_ignore9.yml`：strict-9 的算法意义 no-D-FINE 对照；`use_pre_outputs=false` 会同时删除 D-FINE pre head、`pre_outputs`、`dn_pre_outputs` 和所有 `*_pre` loss，CDN/MAL 保留。
- `..._rf_neck.yml`：EC-compatible RF 三尺度 neck。
- `..._rf_neck_p4_ignore9.yml`：严格 RF P4、IdentityEncoder、single-level decoder、9 类训练。
- `run_ecdet_l_dinov2s_patch16_dec3_liver_dfine_ablation.sh`：统一 runner，按实验名选配置和 test JSON；strict P4 使用 `test_ignore_9_12.json`。
- `submit_ecdet_l_dinov2s_patch16_dec3_liver_dfine_ablations.sh`：创建 5 分钟 waiter，strict P4 固定目标 `cu02:0-3`。
- `wait_ecdet_l_dinov2s_patch16_dec3_liver_tmux.sh`：远端检查 tmux 和空闲 GPU，默认要求每卡至少 28000 MiB free。

当前新增配置和 Slurm：

| 文件 | 用途 |
| --- | --- |
| `ecdetseg/configs/ecdet/ecdet_x_liver_ignore9.yml` | ECDet-X、strict-9、全局 batch 32 |
| `ecdetseg/configs/ecdet/ecdet_x_liver_ignore9_150e.yml` | ECDet-X 150 epochs、patience 30、stop_epoch 148 |
| `ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec5_liver_ignore9.yml` | strict-9 DINOv2-S dec5；只改变 decoder depth |
| `slurm/ecdet_x_liver_ignore9_two_init.sbatch` | 已完成 Job 795；双卡、每卡 16、50 epochs |
| `slurm/ecdet_x_liver_ignore9_two_init_4gpu_150e.sbatch` | 当前 Job 843；四卡、每卡 8、150 epochs |
| `slurm/ecdet_dinov2s_dec5_liver_ignore9_4gpu_150e.sbatch` | 当前 Job 842；DINOv2-S no-register + random dec5 |

所有新 Slurm 都在训练结束后自动选择 `best.pth`，并用 `test_ignore_9_12.json` 做最终 strict-9 test。输出目录存在 `last.pth` 时会 resume；重提任务前必须检查输出目录，防止续跑错误实验状态。

### 4.4 Ignore 数据与评估

- `make_liver_ignore_9_12_annotations.py` 生成严格 9 类 train/valid/test JSON。
- `coco_eval.py` 支持 `ignore_category_ids=[9,10,11,12]`，只用于 evaluator-only ignore。
- `evaluate_ecdet_liver_ignore_9_12.sh` 在原始 13 类 checkpoint 上过滤 GT 和预测后重新评估。
- `ec_engine.py` 增加 COCO PR curve 上的 macro-F1 和 recall-grid 记录。

### 4.5 Solver bug 修复

文件：`ecdetseg/engine/solver/ec_solver.py`

`evaluate()` 同时返回：

```python
{
    "coco_eval_bbox": [...],
    "coco_eval_bbox_f1": float,
    "coco_eval_bbox_f1_recall": float,
}
```

旧代码对所有 `test_stats[k]` 都执行 `enumerate(test_stats[k])`，因此在 epoch 0 评估后遇到 F1 标量时报 `TypeError: 'float' object is not iterable`。现在通过 `_metric_values()` 统一转换为列表；TensorBoard 可记录标量，best checkpoint 只按 `coco_eval_bbox[0]` 选择，early stopping 也使用同一个 COCO 主指标。

## 5. DINOv2 权重与 register 注意事项

当前 YAML 使用的权重路径是：

```text
/cobot/Code/xiangshaochong/checkpoints/dinov2/dinov2_vits14_pretrain.pth
```

用户此前提供过另一个候选权重：

```text
/cobot/Code/CODE/eomt/checkpoints/dinov2/vit_small_patch14_reg4_dinov2.pth
```

两者不能只按文件名互换。接手后必须确认：

1. 是否带 4 个 register token；
2. checkpoint key 是否与 timm `vit_small_patch14_dinov2` 一致；
3. `pos_embed` 是否带 CLS/register 前缀；
4. `patch_embed.proj.weight` 是否能从 patch14 重采样到 patch16；
5. `_load_weights()` 的 strict 检查是否报告 missing/unexpected keys。

当前实验口径是无 register DINOv2-S。register 版本可能影响 token 数、位置编码和中间 feature 语义，不能在没有单独记录的情况下混入 baseline。

## 6. 数据集与类别口径

### 6.1 原始数据

根目录：`/cobot/Data/Lesion_det/det_liver`
图像目录：`/cobot/Data/Lesion_det/det_liver/img`
标注目录：`/cobot/Data/Lesion_det/det_liver/annotations`

原始 JSON 是 13 类。下表格式为“含该类别标注的图像数 / 标注实例数”；类别 7 虽然在 categories 中存在，但当前 split 没有实例。

| ID | 名称 | train | valid | test |
| ---: | --- | ---: | ---: | ---: |
| 0 | 囊性 | 10020 / 11245 | 552 / 577 | 707 / 866 |
| 1 | 结石钙化 | 2027 / 2134 | 349 / 371 | 498 / 557 |
| 2 | 血管瘤 | 2828 / 2896 | 360 / 365 | 540 / 548 |
| 3 | 实性 | 2252 / 2518 | 214 / 241 | 285 / 361 |
| 4 | 胆囊结石 | 4990 / 6291 | 397 / 536 | 602 / 804 |
| 5 | 胆囊息肉 | 3683 / 4175 | 352 / 400 | 539 / 600 |
| 6 | 胆汁淤积 | 399 / 402 | 52 / 52 | 78 / 78 |
| 7 | 胆囊占位 | 0 / 0 | 0 / 0 | 0 / 0 |
| 8 | 积液 | 967 / 969 | 105 / 108 | 143 / 159 |
| 9 | `16` | 20 / 21 | 16 / 16 | 23 / 23 |
| 10 | `17` | 0 / 0 | 0 / 0 | 0 / 0 |
| 11 | `18` | 7 / 7 | 1 / 1 | 3 / 3 |
| 12 | `19` | 1 / 1 | 1 / 1 | 8 / 8 |

原始 split 总量：

| split | 图像数 | 标注数 | categories |
| --- | ---: | ---: | ---: |
| train | 29,540 | 30,659 | 13 |
| valid | 2,504 | 2,668 | 13 |
| test | 3,315 | 4,007 | 13 |

### 6.2 两种 Ignore 必须严格区分

**Evaluator-only ignore（旧 EC 结果）：**

- checkpoint 仍按 13 类训练；
- head 仍为 `num_classes=13`；
- evaluator 过滤 category ID `9,10,11,12` 的 GT 和预测；
- best epoch 仍由原始 13 类 valid 指标确定；
- 这不是重新训练的 9 类模型。

**Strict 9-class training（当前正式实验）：**

- train/valid/test JSON 都删除 category ID `9-12`；
- `num_classes=9`；
- head、matcher、CDN class embedding 从 9 类任务初始化；
- 不能与 13 类训练、评估时过滤的 EC-full 直接当作单变量 neck 对比。

> [!warning] 9 类 head 与 8 个有效评估类别
> 三个 strict JSON 的 `categories` 都声明 ID `0..8`，所以模型 head 是 9 类；但 ID 7“胆囊占位”在 train/valid/test 中均无 GT。它不会获得正样本梯度，COCO evaluator 对该类产生无效值并在均值中跳过。因此当前正式表应描述为“strict-9 模型”，同时承认实际均值由 8 个有 GT 类别贡献。

派生 JSON 当前统计：

| 文件 | 图像数 | 标注数 | categories |
| --- | ---: | ---: | ---: |
| `train_ignore_9_12.json` | 29,540 | 30,630 | 9 |
| `valid_ignore_9_12.json` | 2,504 | 2,650 | 9 |
| `test_ignore_9_12.json` | 3,315 | 3,973 | 9 |

不要把 category name `16/17/18/19` 当成 category ID；当前过滤依据是 JSON 中的整数 ID `9-12`。

## 7. 训练和评估固定口径

除非用户明确要求改变，否则所有当前 liver 消融固定：

| 项目 | 值 |
| --- | --- |
| 输入 | 640x640 |
| patch | 16 |
| DINO blocks | 12 blocks；按实验选择中间输出 |
| decoder | dec3 基线为 3 层；Job 842 是严格单变量 dec5 对照；ECDet-X 为其原生 4 层 |
| queries | normal queries 300 |
| CDN | full 默认 `num_denoising=100`；no-CDN 为 0 |
| epoch | 150 |
| early stop | patience 30，min delta 0 |
| total batch | 32，四卡时通常每卡 8 |
| seed | 42 |
| AMP | 开启 |
| SyncBN / EMA | 开启 |
| optimizer | AdamW，backbone lr 5e-6，其他参数基础 lr 5e-4，weight decay 0.000125 |
| augmentation | EC 官方 schedule；当前基线 mosaic prob 1.0，mosaic/mixup epoch 24，stop epoch 148 |

结果记录规则：

1. `Best epoch` 必须按验证集主指标 `coco_eval_bbox[0]` 选择。
2. F1、mAP@50、mAP@50:95 必须取同一 best epoch，不能分别取各自峰值。
3. 当前 F1 是 IoU=0.50 COCO precision-recall 曲线上的最大 macro-F1，不是固定置信度阈值下的 F1。
4. 参数量要明确统计口径。当前报告的 `Model Params` 使用 `model.deploy()` 后参数；训练态模型因保留 auxiliary 结构而更大，不能混写。
5. 只有一个 seed 时，约 0.5 AP 以内的差异只能作为待复现信号。
6. 严格 9 类训练必须和严格 9 类 EC-full 对照比较；否则只能说明结构和类别口径同时改变。

## 8. 已完成结果与解释边界

### 8.1 Strict-9 消融最终 test

以下结果都来自 strict-9 模型的 `best.pth` 独立 test。`Best Epoch` 是训练过程中 valid mAP50-95 最优 epoch，其余指标取该 checkpoint 的 test 结果。

| 实验 | Deploy Params | Best Epoch | F1 | mAP50 | mAP50-95 | P | R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EC-full | 31,364,873 | 46 | 0.594 | 0.593 | 0.281 | 0.598 | 0.590 |
| no-CDN | 31,362,313 | 53 | 0.594 | 0.589 | 0.270 | 0.570 | 0.620 |
| no-FDR decode | 31,761,300 | 45 | 0.598 | 0.601 | 0.283 | 0.569 | 0.630 |
| no-GO-DDF | 31,364,873 | 46 | 0.592 | 0.592 | 0.282 | 0.574 | 0.610 |
| no-FDR + no-GO-DDF | 31,264,776 | 47 | 0.588 | 0.588 | 0.278 | 0.559 | 0.620 |
| no-Mosaic + no-MAL | 31,364,873 | 47 | 0.587 | 0.582 | 0.263 | 0.605 | 0.570 |
| RF P4 neck | 27,757,881 | 54 | 0.589 | 0.587 | 0.274 | 0.554 | 0.630 |

解释边界：

- 三组 D-FINE 机制消融相对 EC-full 的 mAP50-95 变化仅为 `+0.002/+0.001/-0.003`，单 seed 下影响有限。
- `no-FDR decode` 保留分布监督并增加连续框头，不能解释为完整移除分布回归。
- `no-Mosaic + no-MAL` 下降最大，mAP50-95 降低 0.018。
- RF P4 同时改为单尺度、移除 HybridEncoder，不是 projector-only 对照。
- evaluator-only ignore 的旧表仍在 `EC_ABLATIONS.md`，但不能填入上表。

### 8.2 已完成 ECDet-X 两权重对比（Job 795）

两者均为双卡、每卡 16、全局 batch 32、50 epochs、4-layer ECDet-X，并使用相同 strict-9 数据。分类头和 CDN class embedding 因 80/71 类到 9 类而重新初始化。

| 初始化 | Valid Best Epoch | Test F1 | Test mAP50 | Test mAP50-95 | Test AP75 | Test AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO decoder | 38 | 0.584604 | 0.567 | 0.269 | 0.216 | 0.564 |
| moreOrgan | 42 | 0.579578 | 0.582 | 0.269 | 0.213 | 0.575 |

结果说明：moreOrgan 的 mAP50 `+0.015`、AR100 `+0.011`，但 mAP50-95 持平且 AP75 略低，优势主要是低 IoU 下的召回而不是更精确的定位。moreOrgan 对“胆汁淤积”和“积液”提升明显，但在多个常见类别上略低。当前结果不能证明 moreOrgan 初始化整体优于 COCO decoder。

对应输出：

```text
outputs/ecdet_x_liver9_from_coco_decoder
outputs/ecdet_x_liver9_from_moreorgan
outputs/slurm/ecx-liver9-init-795_0.out
outputs/slurm/ecx-liver9-init-795_1.out
```

## 9. 当前运行任务操作手册

状态快照：2026-08-09，约 epoch 13-14。实时状态必须以 `squeue` 和日志为准。

| Array/Job | 初始化与模型 | 节点/分区 | 输出目录 |
| --- | --- | --- | --- |
| `843_0` | 完整 COCO ECDet-X，4-layer decoder | cu03 / batch | `outputs/ecdet_x_liver9_from_coco_decoder_4gpu_bs8_150e` |
| `843_1` | 完整 moreOrgan ECDet-X，4-layer decoder | cu04 / debug | `outputs/ecdet_x_liver9_from_moreorgan_4gpu_bs8_150e` |
| `842` | DINOv2-S no-register backbone + random 5-layer decoder | cu02 / batch | `outputs/ablation/ecdet_l_dinov2s_patch16_dec5_liver_ignore9_4gpu_bs8_150e` |

共同训练设置：

```text
4 GPUs × 8 samples/GPU = global batch 32
epochs=150
early_stop_patience=30
seed=42
AMP + SyncBN + EMA
Mosaic/MixUp epoch=24
stop_epoch=148
strict train/valid/test JSON
```

监控命令：

```bash
squeue -j 842,843 -o '%.12i %.24j %.10P %.10T %.12M %.4D %R'
tail -f outputs/slurm/ecdino-dec5-4g150-842.out
tail -f outputs/slurm/ecx-liver9-4g150-843_0.out
tail -f outputs/slurm/ecx-liver9-4g150-843_1.out
tail -f outputs/slurm/ecdino-dec5-4g150-842.err
tail -f outputs/slurm/ecx-liver9-4g150-843_0.err
tail -f outputs/slurm/ecx-liver9-4g150-843_1.err
```

接手时必须确认：

1. 三个任务仍为 RUNNING，或若已结束则检查 `sacct -j 842,843` 的 ExitCode。
2. 不要只看当前 epoch 的 F1；best checkpoint 按 valid `coco_eval_bbox[0]` 选择。
3. early stop 后 Slurm 会自动用 `best.pth` 跑 strict test；必须等待第二个 `srun` 完成再宣布任务完成。
4. 若任务被 requeue，脚本会从各自 `last.pth` resume；不要改动原输出目录中的 checkpoint。
5. 当前 5-layer deploy 参数是 `33,949,777`；训练态参数为 `34,199,671`。旧 dec3 报告参数 `31,364,873` 是 deploy 口径。

### 9.1 DINOv2-S dec5 启动验证

Job 842 已实际打印并验证：

- no-register checkpoint 原始 175 tensors；adapter 去掉 `mask_token` 后严格加载 174 tensors；
- patch projection 与位置编码从 patch14/原网格插值到 patch16/40x40；
- 5 个 decoder layers、5 个 score heads、5 个 132-dim FDR heads；
- 所有 5 层 cross-attention 都是 `MSDeformableAttention`；
- 每层 `num_levels=3`、`num_points=[3,6,3]`；
- epoch 0 已完成且无 stderr 错误。

### 9.2 历史 RF P4 状态

RF P4 waiter/tmux 已不是当前待办。该任务已完成训练和 test，正式结果见第 8.1 节及 `EC_ABLATION_REPORT.md`。除非用户明确要求复现，不要重新启动旧 waiter。

## 10. 后续修改与提交规则

当前用户要求不使用 Git，但仍要保持工作树可追溯：

- 不覆盖已有 `outputs/` 目录；新实验使用新名称。
- 不删除旧 checkpoint、日志或 waiter，除非用户明确要求。
- 不杀其他节点上的训练进程；先看 PID、命令行和输出目录归属。
- 改结构前先做单 batch forward、loss backward、1 epoch smoke、参数量检查。
- 改类别数量时同时检查 classifier、DN embedding、matcher、postprocessor、COCO evaluator 和 test annotation。
- 改 feature level 时同时修改 backbone adapter、encoder、decoder 的 channels/strides/levels/points。
- 修改训练脚本时检查 resume、test-only、DDP、AMP 和 `CUDA_VISIBLE_DEVICES`。
- 任何新结果都要写明 checkpoint、best epoch、类别口径、是否 evaluator-only ignore、seed 和节点。

## 11. 接手后的建议顺序

1. 查看 Job `842/843` 的实时状态和 stderr，不要重复提交。
2. 若任务完成，确认训练 `srun` 与最终 strict test `srun` 都是 ExitCode 0，并记录实际 early-stop/best epoch。
3. 从三个 `best.pth` 的最终 test 段读取 F1、mAP50、mAP50-95、P、R；不要使用 `last.pth` 或分别挑选指标峰值。
4. 对比 Job 842 与原 strict-9 dec3，只把差异归因于 decoder depth；使用 `EC_DINOV2_DECODER_3_VS_5.md` 的参数统计口径。
5. 对比 Job 843 两项与已完成 Job 795，分析 150 epochs/patience 30 是否改变两种预训练初始化的结论。
6. 更新 `EC_ABLATION_REPORT.md`、`EC-严格9类消融实验报告.md` 和本 handoff 的正式结果表。
7. 类别 7 无 GT；任何“9 类平均”表述都要注明模型 head 为 9 类、有效评估类别为 8 类。
8. 若用户要求完整 RF decoder，先独立阅读 RF-DETR decoder、query、position encoding 和 box head，不要把当前 EC decoder 称为 RF decoder。
9. 任何新的 backbone/register/patch size/decoder 层数变化都要建立独立配置、独立输出目录和独立结果行。
