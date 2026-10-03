# EC V0–V8 交接记录（mg01-out / debug01）

核对日期：2026-09-28。仓库：`/opt/wanrui/EdgeCrafter`，分支 `codex/ec-v-series`，HEAD `afe652f495dba2f3978508c705bc8d574d1a7f8e`。此分支保留 **cmp5L 专用实现**；本文件记录迁移条件，不把乳腺参数视为新数据集默认值。`EC_V_SERIES.md` 将 V0–V6、V7、V8 分别溯源至历史冻结快照。仓库源码和历史脚本不足以证明完整训练结果；尤其 **V8 完整训练和结果尚需用独立运行产物核验**。

## 实现差异

所有 V 组共用 `scripts/ablation/train_cmp5L_ec_v.py`，以 `--v-arm` 选择逻辑；V0 使用 `ECX0/Y0`，V1–V8 使用 `ECX3/Y1`。V1–V8 的 Y1 在第 50 epoch 起衰减教师 KD，到第 80 epoch 为零。每组有独立的 `ecdetseg/configs/ecdet/ecdet_l_dinov2s_cmp5L_ECV{0..8}.yml`，并不对应九个独立训练入口。

| 组 | 相对 V0 的实际方法 |
| --- | --- |
| V0 | 无教师 KD 的检测基线；没有 KD 校准系数。 |
| V1 | EMA 教师使用特权 NP memory，复用学生查询；末层候选排名 KD（`candidate_ranking_kd`）。 |
| V2 | 四层 decoder 分别匹配并做候选排名 KD，四层损失取平均；教师 replay 额外导出四层 logits 并核对末层一致性。 |
| V3 | 对匹配病灶和远离 GT 的候选集做温度 1 的 KL；教师排名反转的候选集跳过（`candidate_set_kd`）。 |
| V4 | V1 排名 KD 加质量过滤的教师 box KD；要求教师同一 GT 的 IoU ≥ 0.5 且优于学生 ≥ 0.05；box 系数在 epoch 10 单独校准。 |
| V5 | 同时取得特权 NP 与普通 memory 教师 logits，用 NP 相对普通教师的正向 margin 增量加权排名对（`np_increment_ranking_kd`）。 |
| V6 | V1 方法，但特权教师 background weight 从 0.2 调为 0.5。 |
| V7 | V1 KD 加学生双分支特征遮挡干预；共享编码器预测，只把干预分支的 decoder 损失差额计入附加损失。历史 V7 使用 `fix1` 快照。 |
| V8 | V3 候选集 KL 增加固定背景参考 logit `log(候选数)`，记录 reference probability、KL 分解等诊断；方法代码来自 V8 快照，完整训练及指标未在本仓库证实。 |

核心调用链：`train_cmp5L_ec_v.py` → `cmp5L_ec_v_losses.py`、`cmp5L_shared_query_kd.py`、`cmp5L_query_behavior_kd.py`、`train_cmp5L_shared_query_kd.py`、`cmp5L_ec_fullcycle_core.py` → `ecdetseg/engine/solver/ec_solver.py`、`ec_engine.py` 及 ECDet 模型。`ecdetseg/tests/test_cmp5l_ec_v.py` 覆盖组别契约和 V2–V8 部分损失/行为，但不能替代双卡训练验证。

## 训练、校准、选择和评估入口

- **训练**：`scripts/ablation/train_cmp5L_ec_v.py` 的 `main()` / `ECFullSolver.fit()`；参数必须有配置、`--v-arm`、共同初始化 `-r`、manifest、初始化 SHA256、seed 42、AMP 和输出目录。`--smoke` 会执行真实优化更新，不能当作只读检查。历史提交方式见 `slurm/cmp5L_EC_V_one_arm_v1.sbatch`、`cmp5L_EC_V7_fix1.sbatch`、`cmp5L_EC_V8_preflight.sbatch`、`cmp5L_EC_V8_one_arm.sbatch`；它们都指向旧快照和旧集群路径，不能原样在此运行。
- **准备初始化和 manifest**：`scripts/ablation/prepare_cmp5L_ec_fullcycle.py` 生成 seed-42 学生初始化、七个仅训练集的校准 batch 和带哈希的 `manifest.json`。这是写文件且读数据的准备步骤，本次未执行。历史 manifest 固定图像 ID、四类覆盖和空图控制，不适合直接迁移。
- **KD 校准**：训练脚本中的 `calibrate()` / `_calibrate_ranking()` 在复制的学生和 criterion 上，用 manifest batch 的 decoder 第 3 层梯度比定系数，校验 batch 哈希、四类覆盖、教师和 EMA 不变；V4 的 box 系数在 epoch 10 另行校准。新数据集须只用训练集重建 manifest 并重新校准。
- **best EMA**：`ecdetseg/engine/solver/ec_solver.py` 每 epoch 用 EMA 做验证，`_record_eval_best()` 以 bbox COCO mAP50 严格改善为准保存 `best.pth`；`last.pth` 是末轮。`best.pth` 含 `model` 与 `ema`，不能把文件名误当作单独的权重张量。新数据集的验证指标和选择规则须先定稿，各组一致。
- **普通学生推理 / 评估**：`scripts/ablation/evaluate_cmp5L_qbeh_checkpoint.py` → `probe_cmp5L_decoder_internal_tensors.py::_build()`，用 `--checkpoint ... --weights ema` 取 EMA 学生，`model(samples)` 普通 forward 经 postprocessor 和 COCO evaluator；验证 GT 由 dataloader/evaluator 使用，不进入推理 forward。此评估器仍硬编码四类，且会写预测和指标文件。本次未运行。历史 V 脚本对 best、final 分别调用它；`summarize_cmp5L_ec_v.py` → `summarize_cmp5L_ec_fullcycle.py` 审计 100 epoch、初始化血缘、校准、best/final EMA 指标与预测，最后写 `COMPLETED.json`，不能用源码存在代替审计通过。

## 当前 cmp5L 合同与配置链

`ECV0 → ECY0 → ECX3 → ecfull_base → ecdet_l_dinov2s_breast_cmp5L_363_100e_es20 → ecdet_l_breast_cmp5L_base → ecdet_l → (dataset/coco, ecdet)`；V1–V8 把 `ECY0` 换成 `ECY1`，其余包含链相同。合并配置的实际值必须在有相应环境时用 `engine.core.yaml_utils.load_config` 再核实；本次只检查了静态包含关系。

- 数据：`/cobot/Data/Lesion_det/det_breast/img`；训练/验证标注分别为 `.../annotations/Lesion/Ignore_Delete-Image/train.json` 和 `valid.json`。`num_classes: 4`，`remap_mscoco_category: false`；校准代码也直接使用类 ID `0..3`，V8 诊断特别引用类别 3。历史验证审计固定 2975 张图。
- 输入：训练 Resize 和评估尺寸均为 `640×640`；DINOv2-S adapter 把 patch14 调为 patch16，40×40 位置网格。预训练 backbone 路径：`/cobot/Code/CODE/eomt/checkpoints/dinov2/vit_small_patch14_reg4_dinov2.pth`。
- 初始化：共同 `student_init_seed42.pth` 由官方 DINOv2-S backbone 加随机 EC 检测头生成；训练脚本锁定 seed 42、AMP、初始化 SHA256。不要默认换成成熟乳腺检测器。
- 执行：2 GPU × 每卡 16、总 batch 32、梯度累积 1、SyncBN；100 epoch、early stop 0、mosaic/mixup 到 epoch 24、强增强 stop epoch 98。配置来自乳腺实验的优化器与增强参数也需审查。
- 集群：Slurm 脚本固定 `/cobot/Code/wanrui/EdgeCrafter/outputs/ablation/...` 冻结源码、`/cobot/miniforge3/envs/lw-detr/bin/python`、`/opt/slurm/bin/srun`、双 RTX 5090、`/cobot` 数据路径。它们是历史溯源，不是 `/opt/wanrui/EdgeCrafter` 的现成启动器。

## 缺失产物与迁移前须确认

仓库不包含 DINOv2 权重、共同学生初始化、乳腺图像/标注、七个校准 batch 与 manifest、训练 checkpoint（包括 best/last）、预测 JSONL、验证指标、日志和 `outputs/` 运行产物。迁移前须确认新数据集的许可与训练/验证划分、类 ID/类别映射及 ignore 语义、图像尺寸和预处理、backbone 权重及哈希、适合该数据集的共同初始化、优化器和增强/训练预算、硬件与 batch 合同、训练集校准覆盖及系数、验证集指标和统一的 best EMA 规则、推理评估器中的四类和 2975 图限制、Slurm 路径。修改时应另建新数据集配置/入口并重新验证，而不是直接覆盖本分支的 cmp5L 合同。若需声称 V8 训练完成或比较 V8 指标，必须取得独立的完整 checkpoint、100 epoch 日志、验证预测及审计结果并核对哈希。

## 这台主机的只读检查（实际结果）

| 命令（摘要） | 结果 |
| --- | --- |
| `ssh mg01-out 'hostname -f; id; ls -ld /opt/wanrui /opt/wanrui/EdgeCrafter'` | 实际主机 `debug01`，用户 `wanrui`；`/opt/wanrui` 属于该用户、权限 `drwx------`，目标目录起初不存在。 |
| `git ls-remote --exit-code https://github.com/worry-win/EdgeCrafter.git refs/heads/codex/ec-v-series` | 通过，远端为 `afe652f495dba2f3978508c705bc8d574d1a7f8e`；GitHub SSH 公钥认证未通过，因此使用 HTTPS 克隆。 |
| `git rev-parse HEAD; git status --short --branch` | 克隆后 HEAD 与远端一致，分支跟踪 `origin/codex/ec-v-series`，写本文档前工作树干净。 |
| `python3 -B` 解析 `git ls-files '*.py'` 并逐层检查 V0–V8 `__include__` 文件 | 160 个 Python 文件 AST 解析通过；18 个被引用 YAML 文件均存在。此检查不等于 YAML 合并、模型构建或训练通过。 |
| `timeout 30s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:ecdetseg python3 -B -m unittest ecdetseg.tests.test_cmp5l_ec_v` | 未通过：系统 Python 3.12.3 导入测试时 `ModuleNotFoundError: No module named 'torch'`；0 个实际测试用例运行。未安装依赖。 |
| `test -r` 检查历史 Python、srun、backbone、数据、初始化和 manifest 路径 | `/opt/slurm/bin/srun` 可读；历史 Python、backbone、乳腺数据与标注、共享初始化和 manifest 均缺失或不可读。未下载。 |

未执行训练、GPU smoke、评估、数据读取、依赖安装或任何运行任务修改。静态检查不能验证运行环境、数值正确性、V8 完整训练或指标。
