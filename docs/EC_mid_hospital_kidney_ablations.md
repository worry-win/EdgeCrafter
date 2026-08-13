# EC Mid-Hospital Kidney Ablations

## 文档状态

- 当前所有已完成实验统一归为基线，不计入消融结论。
- 基线结果从各实验 `log.txt` 中按验证集最高 `mAP50-95` 选择 epoch，并从同一 epoch 读取全部指标。
- 指标统一使用 `[0, 1]` 小数表示。
- 四组基线已使用各自的 `best.pth` 完成统一重评，补齐 `F1@IoU=0.95` 和 `F1@IoU=0.50:0.95 mean`。
- 新增 F1 指标来自 debug01 重评任务 `113924`；既有 `F1@IoU=0.50`、`mAP50` 和 `mAP50-95` 保留原训练日志中最佳 epoch 的精确值。

## 公共实验设置

| 项目 | 设置 |
| --- | --- |
| 任务 | 肾脏 6 类目标检测 |
| 标注版本 | `v2_260729` |
| 输入尺寸 | `640 x 640` |
| Backbone | DINOv2-S，原始 patch14 权重适配到 patch16 |
| Encoder | HybridEncoder |
| Decoder | ECTransformer |
| 总 batch size | 32 |
| 最大 epoch | 150 |
| Early-stop patience | 30 |
| Seed | 42 |
| 最佳 epoch 选择 | 验证集 `mAP50-95` 最大值 |

## Section 1：基线测试

### Subsection 1：Mid-Hospital Kidney v2_260729（原始图像）

#### 数据集

| Split | 图像数 | 标注框数 | 图像目录 | 标注文件 |
| --- | ---: | ---: | --- | --- |
| Train | 4,997 | 3,976 | `/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images` | `annotations/v2_260729/train.json` |
| Val | 1,198 | 948 | `/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images` | `annotations/v2_260729/val.json` |

Train 和 Val 共用以下 6 个类别：

1. `shen cuo gou liu`
2. `shen ji shui`
3. `shen jie shi`
4. `shen nang zhong`
5. `shen shi zhi mi man xing bing bian`
6. `shen zang e xing zhong liu`

#### 基线结果

| Baseline | Decoder 层数 | 最佳 epoch | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2-S patch16 dec3 | 3 | 77 | 0.651756 | 0.005875 | 0.378952 | 0.679602 | 0.352387 |
| DINOv2-S patch16 dec4 | 4 | 待完成 | 待填 | 待填 | 待填 | 待填 | 待填 |
| DINOv2-S patch16 dec5 | 5 | 47 | 0.669732 | 0.017554 | 0.398467 | 0.713884 | 0.373374 |

### Subsection 2：CSGv2

#### 数据集

CSGv2 对应当前配置中的 `images_CSG_200m`。它与原始图像基线使用完全相同的 `v2_260729` Train/Val 标注、类别和数据划分，只替换图像目录。

| Split | 图像数 | 标注框数 | 图像目录 | 标注文件 |
| --- | ---: | ---: | --- | --- |
| Train | 4,997 | 3,976 | `/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images_CSG_200m` | `annotations/v2_260729/train.json` |
| Val | 1,198 | 948 | `/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images_CSG_200m` | `annotations/v2_260729/val.json` |

#### 基线结果

| Baseline | Decoder 层数 | 最佳 epoch | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2-S patch16 dec3 | 3 | 37 | 0.693674 | 0.008231 | **0.413271** | 0.748159 | **0.393221** |
| DINOv2-S patch16 dec4 | 4 | 待完成 | 待填 | 待填 | 待填 | 待填 | 待填 |
| DINOv2-S patch16 dec5 | 5 | 39 | 0.689514 | 0.008508 | 0.411576 | 0.736274 | 0.385325 |

当前四组基线中，最高 `mAP50-95` 为 CSGv2 dec3 的 `0.393221`。

#### 推理速度

CSGv2 dec3 和 dec5 的最佳 checkpoint 均已完成正式测速。这里统计的是预生成输入 Tensor 已经位于 GPU 后，`cfg.model.deploy()(images)` 的 model-only 前向速度，包含 Backbone、HybridEncoder 和 ECTransformer decoder。

不包含图片读取、resize/normalize、CPU 到 GPU 传输、PostProcessor、阈值过滤、结果回传和绘图。

测速条件：

| 项目 | 设置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| Batch size | 1 |
| 输入 | `1 x 3 x 640 x 640`，预先驻留 GPU |
| 计时 | CUDA Event |
| Warm-up | 每种精度 100 次 |
| 测量 | 每种精度 1,000 次 |
| 裁剪 | 各去掉最快 50 次和最慢 50 次 |
| 有效样本 | 每种精度 900 次 |
| FP32 | TF32 disabled，IEEE FP32 |
| FP16/AMP | FP32 权重 + CUDA autocast FP16 |

| Baseline | 指标 | FP32 | FP16/AMP |
| --- | --- | ---: | ---: |
| CSGv2 DINOv2-S patch16 dec3 | Mean latency (ms) | 11.139 | 11.317 |
| CSGv2 DINOv2-S patch16 dec3 | Throughput (images/s) | 89.77 | 88.36 |
| CSGv2 DINOv2-S patch16 dec5 | Mean latency (ms) | 13.942 | 15.906 |
| CSGv2 DINOv2-S patch16 dec5 | Throughput (images/s) | 71.72 | 62.87 |

重评与测速记录：

| 任务 | 数组项 | 实验 | 结果日志 |
| --- | ---: | --- | --- |
| `113924` | 0 | 原始图像 dec3 重评 | `/opt/wanrui/EdgeCrafter/outputs/reeval/kidney_extended_f1/normal-dec3/metrics-113924_0.log` |
| `113924` | 1 | 原始图像 dec5 重评 | `/opt/wanrui/EdgeCrafter/outputs/reeval/kidney_extended_f1/normal-dec5/metrics-113924_1.log` |
| `113924` | 2 | CSGv2 dec3 重评 | `/opt/wanrui/EdgeCrafter/outputs/reeval/kidney_extended_f1/csgv2-dec3/metrics-113924_2.log` |
| `113924` | 3 | CSGv2 dec5 重评 | `/opt/wanrui/EdgeCrafter/outputs/reeval/kidney_extended_f1/csgv2-dec5/metrics-113924_3.log` |
| `113925` | 0 | CSGv2 dec3 model-only 测速 | `/opt/wanrui/EdgeCrafter/outputs/benchmark/kidney_csgv2_model_only/csgv2-dec3/benchmark-113925_0.log` |
| `113925` | 1 | CSGv2 dec5 model-only 测速 | `/opt/wanrui/EdgeCrafter/outputs/benchmark/kidney_csgv2_model_only/csgv2-dec5/benchmark-113925_1.log` |

## Section 2：消融结果

### 消融实验登记

以下实验不属于当前基线。只有完成训练并按相同验证口径评估后，才能填写结果和形成消融结论。

| ID | 实验 | 相对基线的变化 | 初始化 | FeatAug | 状态 |
| --- | --- | --- | --- | --- | --- |
| A1 | DINOv2-S patch16 dec4（原始图像、CSGv2） | Decoder 从 3/5 层补充为 4 层 | DINOv2-S pretrained | 无 | 待运行 |
| A2 | ECDet-X COCO-decoder CSGv2 | 模型切换为 ECDet-X | `EC-1+2+coco-decoder.pth` | checkpoint 无该分类头 | 待完成 |
| A3 | ECDet-X organ-decoder CSGv2 | 模型切换为 ECDet-X，使用 organ decoder 初始化 | `EC-1+2+organ-decoder.pth` | 必须启用并加载 crop 分类头 | 等待 FeatAug runtime 实现 |

### 消融指标

| ID | 最佳 epoch | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| A2 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| A3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 相对基线变化

消融实验默认与 CSGv2 DINOv2-S patch16 dec3 基线比较。报告绝对值的同时记录差值：

| ID | ΔF1@0.50 | ΔF1@0.95 | ΔF1 mean | ΔmAP50 | ΔmAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 待填 | 待填 | 待填 | 待填 | 待填 |
| A2 | 待填 | 待填 | 待填 | 待填 | 待填 |
| A3 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 指标定义

### F1@IoU=0.50

沿用当前 `bbox-macro-F1@IoU50(PR-curve)` 实现：在 COCO IoU=0.50 的 precision-recall tensor 上，对有效类别的 precision 做 macro average，然后在 COCO recall grid 上计算 F1，报告最大 F1。

### F1@IoU=0.95

使用与 `F1@IoU=0.50` 完全相同的 macro PR-curve 计算方法，但选择 COCO IoU=0.95 的 precision 切片，并在 recall grid 上报告最大 F1。

### F1@IoU=0.50:0.95 mean

分别计算以下 10 个 IoU threshold 的 macro PR-curve 最大 F1：

```text
0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
```

然后取算术平均：

```text
F1 mean = mean(F1@0.50, F1@0.55, ..., F1@0.95)
```

该指标不是 `mAP50-95`，也不能由 `mAP50-95` 换算得到。

## 待办

1. 消融实验完成后，按最高 `mAP50-95` epoch 回填所有同 epoch 指标和相对基线差值。
