# 原肾脏 V3/V4 冻结实现与复现

本分支把历史 `kidney_EC_V_v2/code_snapshot_v2` 纳入 Git。原实现文件逐字节保存；没有套用 KD80 初始化校准器或 ES 早停状态机。其他 V 系列文件作为原快照及共同依赖保留。

## 两个方法

- V3：末层候选集合 KL，EMA NP 教师，严格教师排序筛选，温度1；每图有效集合平均后按全局图像数归一化。
- V4：成对排序＋质量筛选 box KD。box 使用 EC decoder 解码后的归一化 cxcywh `pred_boxes`，不是直接回归 D-FINE 离散分布 logits；仅蒸馏同编号、匹配到GT且满足原教师质量筛选的 query。

教师使用 live 学生 EMA；After-HE NP 背景0.2、GT区域保持、空图identity；共享 detached q0/r0、不重新Top-K。每轮及最终评估均为正常 EMA 推理，GT不进入预测。

## 原训练与校准协议

肾脏六类，原 train/val，seed42，双 RTX5090，全局 batch32，节点内存64G。共同初始化包括预训练DINO骨干与原共同随机初始化学生状态；不得改用已训练V3 best。

100轮，无早停。前10轮正常训练，KD第一次非零时按原31批训练清单无更新校准；检测 criterion 的校准参数为 epoch=10。V3/V4分别使用原分类校准，V4另行进行原box梯度校准。分类目标L3比0.30，box目标0.10。实际系数由新运行计算并锁定，下面是历史参考值，不能硬填替代校准。

| 方法 | 历史分类系数 | 历史box系数 | 历史正常EMA best AP50 |
|---|---:|---:|---:|
| V3 | 0.5558718699 | 无 | 76.56006% |
| V4 | 1.1309202177 | 3.5241249028 | 76.36430% |

Mosaic/MixUp在第1–24轮开启，之后关闭；第99轮按原solver回载best模型、EMA、优化器等状态并收尾两轮。KD前10轮为0、11–20渐增、21–50保持、51–80退火、81–100为0。正常EMA validation AP50选择best，另存final。不要替换成80轮连续训练或ES显式收尾。

初始化SHA256：`c2ac541e0301ba0dbd309c1507e2c0d806e2344adad7707ddd4e7ddc81f7d4dc`。
31批manifest SHA256：`7d207ebabe2521dbd33678b2a49115ee35a02705ff812e017dd3d14ba9cea9cc`。

## 核验与启动

运行 `python scripts/ablation/verify_v3_v4_frozen.py` 可逐字节核验316个原快照文件。外部初始化、manifest、数据和缓存批不纳入Git；入口仍要求这些文件可读并核验原初始化哈希。

启动脚本：`slurm/kidney_EC_V3_V4_frozen_repro.sbatch`。明确设置一个尚不存在的输出根目录，将V3/V4作为两个独立作业提交：

```bash
sbatch --job-name=V3-original-repro --export=ALL,V_ARM=V3,V_REPRO_ROOT=/opt/wanrui/EdgeCrafter/outputs/ablation/kidney_V34_git_repro_v1,V_FROZEN_CODE=$PWD slurm/kidney_EC_V3_V4_frozen_repro.sbatch
sbatch --job-name=V4-original-repro --export=ALL,V_ARM=V4,V_REPRO_ROOT=/opt/wanrui/EdgeCrafter/outputs/ablation/kidney_V34_git_repro_v1,V_FROZEN_CODE=$PWD slurm/kidney_EC_V3_V4_frozen_repro.sbatch
```

启动器仅调整代码与输出路径，原训练器/损失/配置未改。每组先执行原短时AMP/DDP smoke，然后训练、best/final正常EMA validation预测及COMPLETED审计。不会自动运行独立test。输入数据与DINO权重路径仍为原集群路径，迁移前需另行确认；不要修改冻结代码后仍宣称哈希一致。

已提交的复现作业138635/138636仍直接读取原外部快照；本次Git归档不改变它们，也不新增训练。依赖报告138637。完整文件哈希、配置哈希、历史指标和运行环境见 `docs/reproduction/V3_V4_FROZEN.json`。

同seed与协议复现不保证多卡AMP终点逐位一致；不能把单seed validation选模结果当成独立泛化证明。

## 归档验证记录

2026-10-04：316个原快照文件全部哈希一致；启动脚本通过bash语法检查；现有 `test_cmp5l_ec_v.py` 与 `test_kidney_ec_v.py` 共21项CPU测试全部通过（unittest）。此次未启动任何额外训练或推理。
