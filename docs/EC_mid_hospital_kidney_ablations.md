# EC Mid-Hospital Kidney Ablations

## 结果口径

- 本文所有 F1 均使用 WZW-aligned 算法重新计算，不再使用历史日志中的 legacy macro-PR F1。
- 每个实验均加载其已有的 `best.pth` 做 test-only 重评；没有重新训练模型。
- 这些历史 `best.pth` 在训练时按当时的 `mAP50-95` 逻辑选出。当前代码已改为按 `mAP50` 选择 `best.pth`，但本文没有据此重训或重选历史 checkpoint。
- 所有数值来自同一次重评保存的 COCO `precision/scores`，以 `[0, 1]` 小数表示。
- 参数量来自各实验训练日志中的 trainable parameter count，表中换算为百万参数（M）。

## 公共实验设置

| 项目 | 设置 |
| --- | --- |
| 标注版本 | `v2_260729` |
| 输入尺寸 | `640 x 640` |
| Backbone | DINOv2-S，原始 patch14 权重适配到 patch16 |
| Encoder | HybridEncoder |
| Decoder | ECTransformer |
| 总 batch size | 32 |
| 最大 epoch | 150 |
| Early-stop patience | 30 |
| Seed | 42 |
| 历史 checkpoint 选择 | 验证集 `mAP50-95` 最大值 |
| 本次操作 | 加载历史 `best.pth`，使用当前评估代码 test-only 重评 |

验证集标注包含 6 个检测类别，重评时 evaluator `catIds=[0,1,2,3,4,5]`。

## Section 1：基线测试

### Subsection 1：Mid-Hospital Kidney v2_260729（原始图像）

| Split | 图像数 | 标注框数 | 图像目录 | 标注文件 |
| --- | ---: | ---: | --- | --- |
| Train | 4,997 | 3,976 | `/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images` | `annotations/v2_260729/train.json` |
| Val | 1,198 | 948 | `/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images` | `annotations/v2_260729/val.json` |

| Method | Params (M) | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EC, 3-layer decoder | 31.363 | 0.688261 | 0.006949 | 0.412495 | 0.679579 | 0.352346 |
| EC, 5-layer decoder | 33.947 | **0.732372** | **0.016403** | **0.440840** | **0.713856** | **0.373373** |

### Subsection 2：CSGv2

CSGv2 对应配置中的 `images_CSG_200m`。它与原始图像实验使用相同的 `v2_260729` Train/Val 标注、类别和划分，仅替换图像目录。

| Split | 图像数 | 标注框数 | 图像目录 | 标注文件 |
| --- | ---: | ---: | --- | --- |
| Train | 4,997 | 3,976 | `/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images_CSG_200m` | `annotations/v2_260729/train.json` |
| Val | 1,198 | 948 | `/opt/public/wangzhiwei/Ultrasound_Data/COCO_Output/肾脏/images_CSG_200m` | `annotations/v2_260729/val.json` |

| Method | Params (M) | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EC, 3-layer decoder | 31.363 | 0.737563 | 0.009950 | 0.447607 | **0.748113** | **0.393204** |
| EC, 5-layer decoder | 33.947 | **0.745585** | **0.010362** | **0.453677** | 0.736385 | 0.385437 |

#### 推理速度

速度统计对象是预生成输入 Tensor 已位于 GPU 后的 `cfg.model.deploy()(images)` model-only 前向，包含 Backbone、HybridEncoder 和 ECTransformer decoder；不包含图片读取、resize/normalize、CPU 到 GPU 传输、PostProcessor、阈值过滤、结果回传和绘图。

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

| Method | 指标 | FP32 | FP16/AMP |
| --- | --- | ---: | ---: |
| CSGv2 EC, 3-layer decoder | Mean latency (ms) | 11.139 | 11.317 |
| CSGv2 EC, 3-layer decoder | Throughput (images/s) | 89.77 | 88.36 |
| CSGv2 EC, 5-layer decoder | Mean latency (ms) | 13.942 | 15.906 |
| CSGv2 EC, 5-layer decoder | Throughput (images/s) | 71.72 | 62.87 |

## Section 2：CSGv2 消融结果

下面按消融目的组织结果。为使每组能直接比较，同一个对照实验会在不同组中重复出现。

### Group 1：CDN

| Method | Params (M) | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EC | 31.363 | **0.737563** | 0.009950 | **0.447607** | **0.748113** | **0.393204** |
| EC - CDN | 31.361 | 0.736851 | **0.032633** | 0.434049 | 0.747688 | 0.381112 |

### Group 2：D-FINE 主链

| Method | Params (M) | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EC | 31.363 | 0.737563 | 0.009950 | 0.447607 | 0.748113 | **0.393204** |
| EC - GO-LSD | 31.363 | 0.725506 | **0.023908** | 0.441692 | 0.727683 | 0.383905 |
| EC - D-FINE | 31.130 | **0.744597** | 0.023011 | 0.448775 | **0.749416** | 0.392016 |
| Continuous + GO | 31.130 | 0.739551 | 0.019069 | **0.450116** | 0.743755 | 0.389731 |

### Group 3：D-FINE × CDN 组合消融

| Method | Params (M) | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EC | 31.363 | 0.737563 | 0.009950 | 0.447607 | 0.748113 | **0.393204** |
| EC - CDN | 31.361 | 0.736851 | **0.032633** | 0.434049 | 0.747688 | 0.381112 |
| EC - D-FINE | 31.130 | **0.744597** | 0.023011 | **0.448775** | **0.749416** | 0.392016 |
| EC - D-FINE - CDN | 31.128 | 0.701758 | 0.015417 | 0.423236 | 0.707381 | 0.367991 |

### Group 4：Mosaic × 分类损失

| Method | Params (M) | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mosaic + MAL（EC） | 31.363 | 0.737563 | 0.009950 | 0.447607 | **0.748113** | **0.393204** |
| Mosaic + Focal | 31.363 | **0.746001** | **0.024640** | **0.450944** | 0.737256 | 0.377110 |
| No Mosaic + Focal | 31.363 | 0.726055 | 0.019412 | 0.439303 | 0.733912 | 0.376598 |

### Group 5：Decoder depth × D-FINE

| Method | Params (M) | F1@IoU=0.50 | F1@IoU=0.95 | F1@IoU=0.50:0.95 mean | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EC, 3-layer decoder | 31.363 | 0.737563 | 0.009950 | 0.447607 | 0.748113 | **0.393204** |
| EC - D-FINE, 3-layer decoder | 31.130 | 0.744597 | **0.023011** | 0.448775 | **0.749416** | 0.392016 |
| EC, 5-layer decoder | 33.947 | **0.745585** | 0.010362 | **0.453677** | 0.736385 | 0.385437 |
| EC - D-FINE, 5-layer decoder | 33.649 | 0.737065 | 0.014291 | 0.452148 | 0.739542 | 0.387785 |

### 当前最优单项

在全部已完成的 CSGv2 实验中，各列最高值不来自同一个方法：

| 指标 | Method | 数值 |
| --- | --- | ---: |
| F1@IoU=0.50 | Mosaic + Focal | 0.746001 |
| F1@IoU=0.95 | EC - CDN | 0.032633 |
| F1@IoU=0.50:0.95 mean | EC, 5-layer decoder | 0.453677 |
| mAP50 | EC - D-FINE | 0.749416 |
| mAP50-95 | EC, 3-layer decoder | 0.393204 |

## 指标定义

### WZW-aligned F1

对每个 IoU threshold 和每个类别，分别在 COCO recall grid 上寻找该类别 F1 最大的 PR 点。不同类别可以选择不同的 recall、precision 和 confidence。

类别汇总时只保留 `class_best_f1 > 0` 的类别，然后计算：

```text
mean_P = mean(class_best_precision)
mean_R = mean(class_best_recall)
overall_F1 = 2 * mean_P * mean_R / (mean_P + mean_R)
```

因此，整体 F1 不是 `mean(class_best_f1)`。`class_best_f1 > 0` 的过滤规则与目标算法一致，但可能排除完全失败的类别，使总体 F1 偏乐观。本次各实验在 IoU=0.50 时 6 个类别均有效；更高 IoU 下仍按该规则逐 threshold 过滤。

- `F1@IoU=0.50`：IoU=0.50 时的 `overall_F1`。
- `F1@IoU=0.95`：IoU=0.95 时的 `overall_F1`。
- `F1@IoU=0.50:0.95 mean`：IoU 0.50、0.55、...、0.95 共 10 个 `overall_F1` 的算术平均。
- `mAP50` 和 `mAP50-95`：同一份 COCO precision tensor 上按标准 COCO 类别均值计算，不沿用 F1 的 `class_best_f1 > 0` 过滤。

历史字段 `coco_eval_bbox_f1` 仍保留 legacy macro-PR 语义，但本文不使用该字段填表。

## 重评记录

统一重评输出目录：`outputs/reeval/kidney_wzw/`。每个方法目录包含 `metrics.log` 和 `eval.pth`；表中精确结果由 `eval.pth` 的 COCO `precision/scores` 计算，日志用于确认 checkpoint 加载和运行完整性。

以下尚未完成，因此没有混入结果表：原始图像/CSGv2 的 4-layer decoder、ECDet-X COCO-decoder CSGv2、ECDet-X organ-decoder CSGv2。
