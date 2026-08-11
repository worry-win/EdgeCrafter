# EdgeCrafter 消融 2.0

本文记录当前 EdgeCrafter 检测分支中 CDN、FDR、FGL、DDF 和 GO union 的实现边界，以及三组既有消融配置的真实含义。本文以仓库代码为准，主要对应 strict-9 liver 配置：

```text
train_ignore_9_12.json
valid_ignore_9_12.json
test_ignore_9_12.json
```

代码中使用的术语是 **FDR**（Fine-grained Distribution Refinement）。如果其他材料中写作 FDL，本文统一按当前代码的 FDR 命名。

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

## 二、D-FINE / GO-DDF 三组消融

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
  -> 合并 final/aux/pre/encoder 的 Hungarian matching
  -> 为 box/localization loss 提供跨层正样本集合
```

因此：

- GO 可以独立关闭，因为它是匹配集合策略；
- DDF 不能在完全删除 distribution 后保留，因为 DDF 的 student/teacher 都是 distribution；
- FGL 也依赖 `pred_corners`；
- LQE 依赖 distribution quality，因此 distribution 删除时一并关闭。

### 2.2 三组配置总览

| 实验 | FDR/FGL | DDF | GO union | pre path | CDN | 结论口径 |
|---|---|---|---|---|---|---|
| `no_go_ddf` | 保留 | 删除 | 删除 | 保留 | 保留 | FDR/FGL 保持，组合去掉 GO 与 DDF |
| `no_fdr_no_go_ddf` | 删除 | 删除 | 删除 | 仍保留 | 保留 | continuous box + no FDR/FGL/DDF/GO，但不是完整 no-D-FINE |
| `no_dfine_ignore9` | 删除 | 删除 | 删除 | 删除 | 保留 | strict no-D-FINE，对照 CDN 保持开启 |

下面分别说明各组的配置、loss、实现方式和可归因问题。

### 2.3 `no_go_ddf`

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

`use_uni_set=false` 后，`loss_bbox`、`loss_giou` 和 `loss_fgl` 使用各自输出分支的局部 Hungarian matching，不使用跨 final/aux/pre/encoder 的 union。当前 criterion 仍会构造 `indices_go`，但关闭后它不再传入这些 loss；这只是冗余索引计算，不会产生模型梯度或改变训练结果。

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

它可以反映 GO 与 DDF 组合对训练监督和中间层优化的贡献。

#### 不能说明的问题

它不能单独归因于 GO，也不能说明 FDR 是否重要，因为 FDR 和 FGL 仍然存在。由于 GO 与 DDF 同时关闭，结果也不是一个严格的单因素 GO 消融。

### 2.4 `no_fdr_no_go_ddf`

配置文件：

```text
ecdetseg/configs/ecdet/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_no_go_ddf_ignore9.yml
```

关键开关：

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

#### 删除内容

因为 `use_aux_distribution=false`，以下模块不会创建或执行：

```text
dec_bbox_head（4 × 33 distribution head）
Integral
up / reg_scale
pred_corners / ref_points distribution path
FDR box decoding
LQE
FGL
DDF
```

decoder 使用连续回归 head：

```python
box_i = sigmoid(
    continuous_bbox_head[i](output_i)
    + inverse_sigmoid(reference_i)
)
reference_{i+1} = box_i.detach()
```

GO 也通过 `use_uni_set=false` 从 box/local loss 的匹配路径中移除。

#### 仍然保留的内容

该配置没有设置 `use_pre_outputs=false`，所以仍然保留：

```text
pre_bbox_head
pre_outputs
dn_pre_outputs
pre MAL/L1/GIoU loss
```

因此它不是完整 no-D-FINE，而是：

```text
continuous-box decoder
+ no FDR/FGL/DDF/GO
+ D-FINE pre auxiliary path
+ CDN
+ MAL
```

#### 产生的 loss

criterion 的 `losses` 只有：

```text
mal
boxes -> loss_bbox + loss_giou
```

所以不会出现：

```text
loss_fgl*
loss_ddf*
```

但因为 pre path 仍在，仍会出现：

```text
loss_mal_pre
loss_bbox_pre
loss_giou_pre
loss_mal_dn_pre
loss_bbox_dn_pre
loss_giou_dn_pre
```

CDN 仍然为每个 decoder 层产生 `loss_mal_dn_i`、`loss_bbox_dn_i`、`loss_giou_dn_i`。

#### 能说明的问题

该实验测量：

> 在 ECTransformer 中使用连续 4D box regression、同时移除 FDR/FGL/DDF/GO 后，保留 D-FINE pre auxiliary path 和 CDN 时的性能。

它比 `no_fdr_decode` 更接近 distribution-free decoder，但不能将结果表述为“完整移除 D-FINE”。

### 2.5 `no_dfine_ignore9`

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

#### 相对 `no_fdr_no_go_ddf` 的额外删除

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

> 在保留 EC attention、MAL、CDN 和基本 box losses 的前提下，完整移除 D-FINE 的 FDR distribution、FGL、DDF、LQE、GO union以及 pre auxiliary path 后，模型性能如何变化。

它是 D-FINE family 的完整算法对照，但仍不是 RF-DETR decoder，也不能直接解释成 RF baseline。

### 2.6 三组实验的严格归因边界

三组实验不是完全正交的四格 factorial 设计：

```text
no_go_ddf          = FDR/FGL 保留，GO + DDF 删除
no_fdr_no_go_ddf   = FDR/FGL/GO/DDF 删除，pre 保留
no_dfine_ignore9   = FDR/FGL/GO/DDF/pre 全部删除
```

因此应按以下方式解读：

1. `no_go_ddf` 与 EC-full 的差异，反映 GO+DDF 组合的贡献，不能单独拆成 GO 或 DDF 的贡献。
2. `no_fdr_no_go_ddf` 与 `no_go_ddf` 的差异，同时混合了 FDR/FGL 删除和 continuous box regression 替换，且还涉及 pre path 的相对影响，不能只归因于 FDR。
3. `no_dfine_ignore9` 与 `no_fdr_no_go_ddf` 的差异，主要用于识别 D-FINE pre head/pre loss 的额外贡献。
4. `no_dfine_ignore9` 与 EC-full 的差异，才是当前代码下最完整的 D-FINE family 对照。

如果要做更严格的独立归因，建议额外实现两组：

```text
EC - GO：只设置 use_uni_set=false，保留 FDR/FGL/DDF
EC - FDR-family + GO：删除 distribution/FGL/DDF/LQE，保留 use_uni_set=true
```

注意，原始 DDF 依赖 FDR distribution，因此“完全删除 FDR、同时保留原始 DDF”在当前定义下不可实现；除非另行设计连续 box distillation loss，但那就不再是原始 DDF。

### 2.7 运行时消融审计

正式训练或报告结果前，应从首个 batch 的日志和 runtime output 检查 loss key：

| 配置 | 应存在 | 不应存在 |
|---|---|---|
| `no_go_ddf` | `loss_fgl*`、普通/pre/DN MAL/L1/GIoU | `loss_ddf*` |
| `no_fdr_no_go_ddf` | 普通/pre/DN MAL/L1/GIoU | `loss_fgl*`、`loss_ddf*` |
| `no_dfine_ignore9` | final/aux/enc/DN 的 MAL/L1/GIoU | `loss_fgl*`、`loss_ddf*`、所有 `*_pre` |

同时检查模型结构：

```text
no_fdr_no_go_ddf / no_dfine_ignore9 不应创建 dec_bbox_head、Integral、LQE
no_dfine_ignore9 还不应创建 pre_bbox_head
三组 CDN-on 配置都应存在 DN outputs 和 *_dn_* loss
```

这样可以区分：

```text
模块未创建
模块创建但只用于辅助监督
模块仍前向但输出未进入 loss
```

只有第一种和第二种在实验归因上是明确的；第三种通常只是无效计算，应避免作为正式消融定义。
