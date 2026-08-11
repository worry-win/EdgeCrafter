# EC 关闭 D-FINE 与 RF-DETR 严格对齐说明

> 更新时间：2026-08-11
>
> 本文只定义“将当前 EdgeCrafter 检测器改为与指定 RF-DETR liver baseline 对齐”的结构、loss 和训练口径，不代表该模型已经实现或提交训练。实施前必须以本文的目标表和实际 forward/loss 测试为准。

## 1. 对齐目标与参考实现

这里的 RF 不是泛指任意 RF-DETR 配置，而是用户指定的已完成 liver baseline：

```text
/cobot/Code/xiangshaochong/MY_output/202608_1/
rfdetr_ablation_enc(dinov2_s)_dec(rand)/rfdetr_baseline_liver
```

其实际生效参数记录在 `CONFIGURATION.md` 与 `logs/direct-launch.log`，源码入口为：

```text
/cobot/Code/xiangshaochong/rf-detr-develop/
  src/rfdetr/models/lwdetr.py
  src/rfdetr/models/transformer.py
  src/rfdetr/models/criterion.py
  src/rfdetr/models/matcher.py
  scripts/train/run_liver_rfdetr_ablation.py
```

该参考实验的关键设置是：DINOv2-S encoder-only pretraining、640、patch16、12 blocks、P4-only RF projector、3-layer RF/LW-DETR decoder、300 queries、two-stage、`bbox_reparam=true`、`lite_refpoint_refine=true`、`group_detr=1`。它关闭 denoising、D-FINE FGL、D-FINE DDF、DEIMv2 MAL、EC neck 和 EC augmentation。

“严格对齐 RF”有两个层级，不能混用：

| 层级 | 定义 | 是否能只改 YAML |
| --- | --- | --- |
| RF-style loss ablation | 删除 EC 的 FDR/FGL/DDF/GO/CDN/MAL，只保留与 RF 相同的 IA-BCE + L1 + GIoU 监督 | 否，EC 目前没有 IA-BCE loss |
| Full RF detector alignment | 除 loss 外，改为 RF 的 P4 neck、RF Transformer、query 初始化、box parameterization、共享 head、two-stage 及 auxiliary 输出语义 | 否，必须替换/移植 decoder 与 criterion |

仅把 `loss_fgl`、`loss_ddf` 设为零不属于任一种严格对齐。

## 2. 原 EC Decoder 如何使用 D-FINE 策略

当前 liver EC-full 使用 `ECTransformer`，代码在 `ecdetseg/engine/edgecrafter/decoder.py`，criterion 在 `ecdetseg/engine/edgecrafter/criterion.py`。其数据流为：

```text
P3/P4/P5 -> EC HybridEncoder -> 8,400 memory tokens
  -> encoder class/bbox head -> top-300 encoder-memory queries
  -> EC TransformerDecoder (3 layers)
  -> per-layer class head + 132-dim distribution bbox head + LQE
  -> final / auxiliary / pre / encoder / CDN outputs
```

### 2.1 FDR：离散边界分布解码

每个 decoder layer 的 `dec_bbox_head[i]` 输出：

```text
4 edges x (reg_max + 1) = 4 x 33 = 132 logits
```

其中 `reg_max=32`。`Integral` 对每条边的 33-bin 分布取期望，`distance2bbox()` 再以当前 reference box 为基准得到新的 box。后续 layer 使用前一层 box 的 detach 结果作为 reference。最终检测框默认来自该 FDR 分布，不是普通 4 维 MLP 的直接输出。

### 2.2 FGL、DDF、GO union 与 LQE

| EC 机制 | 实际作用 |
| --- | --- |
| FGL | 对 matched query 的四条边分布计算 Fine-Grained Localization loss。 |
| DDF | 以最终 decoder layer 的边界分布为 teacher，对 earlier decoder layers 的边界分布做 KL distillation。 |
| GO union | 汇总 final decoder、aux decoder、pre head、encoder head 的 Hungarian matches；当 `use_uni_set=true` 时，box/local loss 使用 union match，而不是各输出自己的 match。它不是单独的标量 loss。 |
| LQE | 读取边界分布，重标定分类分数。它是 decoder 预测路径，不是 criterion loss。 |

EC-full 的 criterion 为：

```yaml
losses: ['mal', 'boxes', 'local']
weight_dict:
  loss_mal: 1
  loss_bbox: 5
  loss_giou: 2
  loss_fgl: 0.15
  loss_ddf: 1.5
use_uni_set: true
use_fgl: true
use_ddf: true
```

`mal` 是 EC/DEIM 风格的 Modulated Adaptive Loss：正样本分类 target 是 matched box IoU 的 gamma 次幂，负样本权重由 detach 后预测概率构造。它不是 RF baseline 使用的 IA-BCE。

此外，EC-full 仍有 `num_denoising=100` 的 CDN。它会创建 DN query、DN attention mask、DN class embedding，并为 DN decoder layers 和 DN pre head 计算相同种类的监督。

## 3. 当前“关闭 D-FINE”的实际行为

现有配置是：

```text
ecdetseg/configs/ecdet/
ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_no_go_ddf.yml
```

其覆盖内容为：

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

### 3.1 该配置确实删除了什么

| 项目 | 当前 no-FDR/no-GO-DDF 行为 |
| --- | --- |
| FDR 132-dim `dec_bbox_head` | 不创建。 |
| `pred_corners`、Integral、`distance2bbox` 分布解码 | 不执行。 |
| FGL / DDF | 不在 `losses` 中，因此不计算 `loss_fgl` / `loss_ddf`。 |
| LQE | 不创建 `lqe_layers`，分类分数不再由 distribution 校正。 |
| GO union | 关闭；每个输出独立做 Hungarian matching。 |

decoder box 改为每层一个独立 `continuous_bbox_head[i]`，以 `sigmoid(MLP_i(query) + inverse_sigmoid(reference))` 得到 4 维 box。

### 3.2 该配置仍保留什么

| 保留项 | 与 RF baseline 的关系 |
| --- | --- |
| `ECTransformer` 本体 | 不同。EC 使用自己的 decoder layer、输入投影、query-pos MLP 和三尺度 memory 接口。 |
| P3/P4/P5 + HybridEncoder | 不同。指定 RF baseline 是 RF P4-only projector，不使用 EC HybridEncoder。 |
| 8-head self-attention、3-level MSDeformableAttention、`num_points=[3,6,3]` | 不同。RF baseline 为 self-attention 8 heads、cross-attention 16 heads、单 P4、每 head 2 points。 |
| encoder top-k memory 作为 query content | 不同。RF two-stage 的 query content 来自 learned `query_feat`；encoder proposal 主要提供/调制 reference points。 |
| 每层独立分类 head、每层独立连续 bbox MLP | 不同。RF 用一个共享 `class_embed` 和一个共享 `bbox_embed` 对所有 decoder layer 输出做预测。 |
| `pre_bbox_head` 与 `pre_outputs` | 不同。RF baseline 没有 EC 独立 pre loss 分支。 |
| `enc_aux_outputs` | 形式相近但接口不同。RF two-stage 有一个 `enc_outputs` 监督分支；EC 有自己的 top-k encoder auxiliary 输出。 |
| CDN，`num_denoising=100` | 不同。RF baseline `use_denoising=false`。 |
| `loss_mal` | 不同。RF baseline 关闭 DEIMv2 MAL。 |
| 训练策略 | 不同。当前 EC 使用 seed42、global batch32、EMA、SyncBN、patience30 和 EC Mosaic/MixUp；RF baseline 使用 seed20260801、global batch180、无 EMA、无 early stopping、默认 RF augmentation。 |

因此当前实验的准确名称应是：

> **EC decoder 的 continuous-box, no-FDR/no-FGL/no-DDF/no-GO ablation**。

不能称为“RF decoder”或“与 RF baseline 完全对齐”。

### 3.3 当前 no-FDR/no-GO-DDF 仍优化的 loss

目前只有以下梯度 loss：

```text
loss_mal  (weight 1)
loss_bbox (weight 5)
loss_giou (weight 2)
```

在 dec3 + CDN-on 的情况下，它们会施加到：final decoder、2 个 decoder auxiliary layers、`pre_outputs`、1 个 encoder auxiliary output、3 个 DN decoder layers、DN pre output。即最多 9 组 prediction output，每组 3 个可训练 loss。`loss_mask_*` 不在 `losses` 中；cardinality 也不是当前 EC criterion 的训练 loss。

## 4. RF baseline 实际 decoder 和 loss

参考 RF baseline 的核心不是 D-FINE decoder，而是 RF/LW-DETR decoder。其执行路径为：

```text
DINOv2-S blocks 3/6/9/12
  -> RF MultiScaleProjector(P4 only)
  -> one P4 feature [B, 256, 40, 40]
  -> RF Transformer
     learned query_feat + learned refpoint_embed
     two-stage encoder proposals
     3 decoder layers
  -> shared class_embed + shared bbox_embed
  -> final prediction + 2 decoder auxiliary outputs + enc_outputs
```

目标 RF decoder 参数：

| 项目 | RF baseline 值 |
| --- | ---: |
| feature levels | 1, P4 only |
| decoder layers | 3 |
| normal queries / select | 300 / 300 |
| self-attention heads | 8 |
| deformable cross-attention heads | 16 |
| cross-attention points | 2 |
| `group_detr` | 1 |
| `two_stage` | true |
| `bbox_reparam` | true |
| `lite_refpoint_refine` | true |
| Denoising | false |
| D-FINE FGL/DDF | false / false |
| DEIMv2 MAL | false |
| segmentation | false |

### 4.1 RF 的连续 box 表达

RF 的共享 `bbox_embed` 输出 4 维 delta。`bbox_reparam=true` 时：

```text
new_center = delta_xy * reference_wh + reference_xy
new_size   = exp(delta_wh) * reference_wh
```

这与 EC 当前 no-FDR 的 `sigmoid(delta + inverse_sigmoid(reference))` 不同。因此即使都只输出 4 维 box head，也不能视为同一种 decoder 回归路径。

### 4.2 RF 的分类和 criterion

参考 baseline 开启 `ia_bce_loss=true`，关闭 `use_deimv2_mal_loss`。训练 gradient loss 为：

| RF key | 系数 | 含义 |
| --- | ---: | --- |
| `loss_ce` | 1 | IoU-aware Binary Cross Entropy。matched class 的权重/target 由 IoU 与预测概率构造，但公式不是 EC MAL。 |
| `loss_bbox` | 5 | L1 box loss。 |
| `loss_giou` | 2 | GIoU loss。 |

`cardinality_error` 只记录，不反向传播。RF `aux_loss=true` 时会对 2 个中间 decoder output 和 one two-stage `enc_outputs` 重复计算同类监督；没有 EC 的 `pre_outputs`，也没有 DN loss。

RF matcher 仍为每个输出独立 Hungarian matching，权重 `cost_class=2`、`cost_bbox=5`、`cost_giou=2`。这与关闭 GO union 后的“独立匹配”原则相同，但 EC 与 RF 的分类 logits/loss 公式不同，不能只复用 EC `loss_mal`。

RF build path 用 `args.num_classes + 1` 构造检测 head。因此这个 liver reference 在 `num_classes=9` 时实际输出 10 个 classification logits。其 sigmoid postprocess 会将全部 10 个通道展平后取 top-k；第 10 个通道在 IA-BCE 中没有正样本标签，训练时作为恒定负通道。当前 EC strict-9 head 是 9 logits；严格复现必须保留 RF 的 10-logit shape、criterion 和 top-k 行为，不能自行过滤第 10 通道。若改为过滤它，应单独记录为后处理修正，不能称为逐项 RF 复现。

## 5. 从 EC 到完整 RF 对齐必须删除或替换的内容

下表是实现 gate。只有所有“必须”项完成并通过验证，结果才能命名为 `EC + full RF decoder` 或 `full RF detector alignment`。

| 域 | 当前 EC no-FDR/no-GO-DDF | 严格 RF 目标 | 必须操作 |
| --- | --- | --- | --- |
| 输入特征 | 通常 P3/P4/P5，经 HybridEncoder | RF projector 单 P4 | 使用 strict RF P4 neck；移除 HybridEncoder，保持 `IdentityEncoder`。 |
| Decoder class | `ECTransformer` | RF `Transformer` / `TransformerDecoder` | 新建/移植 `RFTransformerDecoder`，不能仅给 `ECTransformer` 加开关。 |
| Cross attention | 3 levels, `[3,6,3]`, EC nhead=8 | 1 level, CA nhead=16, 2 points | 使用 RF `MSDeformAttn` 及其参数、positional encoding 和 mask 语义。 |
| Query 初始化 | encoder top-k memory 作 query content | learned `query_feat` + learned `refpoint_embed`，two-stage proposal 调制 reference | 删除 EC top-k query-content 路径，接入 RF two-stage 初始化。 |
| Box head | 独立 `continuous_bbox_head[i]` | shared RF `bbox_embed` | 删除连续 EC box head module list，使用共享 MLP 和 RF `bbox_reparam`。 |
| 分类 head | 每层独立 `dec_score_head[i]`，9 logits | shared RF `class_embed`，10 logits | 删除 per-layer EC class heads，使用共享 RF head，并同步 output dimension。 |
| FDR 组件 | 当前已关闭 | 不存在 | 保持删除 `dec_bbox_head`、Integral、`pred_corners`、LQE、FGL、DDF。 |
| EC pre head | `pre_bbox_head` + `pre_outputs` | 不存在 | 删除该输出及其 criterion loss。 |
| Encoder supervision | EC `enc_aux_outputs` | RF `enc_outputs` | 替换为 RF two-stage encoder output 与独立 Hungarian supervision。 |
| CDN | 当前仍开启 | 关闭 | `num_denoising: 100 -> 0`，删除 DN embedding/query/mask/output/loss。 |
| 分类 loss | EC `loss_mal` | RF `loss_ce` IA-BCE | 在 criterion 中实现/移植 RF IA-BCE；不得以 EC MAL、VFL 或 focal 代替。 |
| Matching | EC matcher + optional GO | RF Hungarian，independent per output | `use_uni_set=false`，并确认 matcher cost 与 RF logits 语义一致。 |
| Auxiliary losses | final + aux + pre + encoder + DN | final + 2 aux + one enc | 保留 decoder aux 与 encoder aux，移除 pre 与 DN。 |
| Postprocess | EC foreground-only 9-logit 规则 | RF 对 10 个 sigmoid logits 展平 top-k | 逐项复现 RF score selection；不要擅自过滤额外通道。过滤属于另一个后处理变量。 |

## 6. 推荐实施顺序

不要直接在 `ECTransformer` 上拼接大量 `if rf_mode`。那会使 EC 和 RF 两条路径互相污染，也无法验证“删除”是否真实发生。推荐独立注册模型组件：

1. 新增 `RFTransformerDecoder` 与 `RFCriterion`，保留现有 `ECTransformer` / `ECCriterion` 默认行为不变。
2. 新增 full-RF YAML：使用 strict RF P4 backbone path、`IdentityEncoder`、single P4、3 RF decoder layers、9 foreground classes，并使用独立 output directory。
3. 先实现 RF continuous box、shared heads、two-stage initialization、aux/encoder outputs；此时完全不创建 FDR/LQE/pre/CDN modules。
4. 接入 RF IA-BCE、L1、GIoU 和 independent Hungarian matching；保证 weight/cost 分别为 `1/5/2` 与 `2/5/2`。
5. 将 postprocessor 和 class-index mapping 对齐 RF 的 extra no-object logit。
6. 做单 batch forward、criterion backward、DDP one-epoch smoke、参数统计和 `state_dict` key audit。
7. 运行 150 epoch 正式任务前，输出一份 runtime audit：feature shape、query shape、logit shape、每个 loss key、DN/pre/FDR key 均应不存在。

### 6.1 最低验收标准

full-RF mode 的训练 batch 中应当满足：

```text
feature levels                   == 1
P4 feature                       == [B, 256, 40, 40] for 640 input
decoder layers                   == 3
decoder prediction logits        == [B, 300, 10]
auxiliary decoder outputs        == 2
encoder outputs                  == 1
loss keys with gradient          == loss_ce, loss_bbox, loss_giou and RF aux/enc suffixes
pred_corners / loss_fgl/ddf      absent
pre_outputs / *_pre              absent
dn_meta / *_dn                   absent
loss_mal                         absent
```

### 6.2 训练协议也要区分

如果实验目标是**结构和 loss 对齐**，可以保留当前 EC 的 strict-9 JSON、seed42、batch32、150 epoch、patience30、AMP/EMA/SyncBN 和 EC augmentation；但结果名称必须是“RF architecture/loss under EC training protocol”。

如果实验目标是**端到端复现指定 RF baseline**，还必须同步其训练协议：seed `20260801`、四卡 global batch `180`、AdamW `lr=1e-4` / encoder `1.5e-4`、step LR at 100、无 warmup、无 EMA、无 early stopping、默认 RF augmentation、无 Mosaic/MixUp、clip norm `0.1`、BF16 AMP、`sync_bn=false`。这会同时改变大量训练变量，不能与当前 EC-full 作为单变量 decoder 消融。

## 7. 命名与结果解释规则

| 名称 | 合法条件 |
| --- | --- |
| `EC - FDR - GO-DDF` | 仅指当前 continuous-box EC decoder 消融。 |
| `EC + strict RF P4 neck` | 仅 P4 neck/interface 对齐；decoder 仍是 EC，不能称 RF decoder。 |
| `RF-style loss ablation` | 已移除 FDR/FGL/DDF/GO/CDN/MAL，并使用 IA-BCE，但可仍保持 EC Transformer。 |
| `EC + full RF decoder` | 已满足第 5、6 节的架构和 loss 验收项。 |
| `RF baseline reproduction` | full RF decoder 之外，训练协议、数据过滤和初始化也对齐指定 RF baseline。 |

在上述实现完成前，不应将当前 no-FDR/no-GO-DDF 的 `mAP50:95=0.278` 归因成“RF decoder 性能”。它只测量 EC 现有 decoder 在关闭 D-FINE 分布策略后的结果。
