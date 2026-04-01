# TacVerse Benchmark — 最小实验方案

> 目标：用最少实验支撑 4 个核心结论
> 策略：C1–C3 共用 1 个 backbone + 固定 target；C4 独立做 representation study

---

## 核心结论 → 实验映射

| 结论 | 内容 | 所需对比 |
|------|------|---------|
| C1 | 单传感器内可学，跨传感器显著下降 | within vs cross (同一 target) |
| C2 | 多源联合 > 单源，但仍有 gap | cross vs loso vs within |
| C3 | 少量标注即可恢复，知识"可迁移但未对齐" | fewshot curve (k↑ → acc↑) |
| C4 | MAE 自监督提供统一表征 | 5 model 对比 (独立 study) |

---

## Part A — 主结论 C1–C3

**统一配置：**
- Backbone: `vit_small_patch16_224 / imagenet`（单一 backbone，排除模型差异干扰）
- 每个任务固定 **1 个 target sensor**
- Grating 仅保留 `pattern` variant
- Shape 仅保留 `longtail` 版本

### A1. Force Regression → target: MagicGripper

> 选择理由：磁性触觉传感器，中等性能，跨传感器落差明显

| # | Protocol | Source | Target | 支撑 | 数量 |
|---|----------|--------|--------|------|------|
| 1 | within_sensor | MagicGripper | MagicGripper | C1 上界 | 1 |
| 2–7 | cross_sensor | 其余 6 sensor 各 1 | MagicGripper | C1 下降 + C2 单源 | 6 |
| 8 | leave_one_sensor_out | 其余 6 sensor 联合 | MagicGripper | C2 多源 | 1 |
| 9–12 | fewshot_adaptation | 联合 + k-shot MG | MagicGripper | C3 恢复 | 4 |
| | | | | **小计** | **12** |

> fewshot ratios: 0.005, 0.01, 0.05, 0.1

### A2. Grating Classification (pattern) → target: GelsightMarker

> 选择理由：最经典的 GelSight 传感器，论文 motivation 常用

| # | Protocol | Source | Target | 支撑 | 数量 |
|---|----------|--------|--------|------|------|
| 1 | within_sensor | GelsightMarker | GelsightMarker | C1 上界 | 1 |
| 2–4 | cross_sensor | GelsightNoMarker, MagicGripper, ViTacTip 各 1 | GelsightMarker | C1 + C2 | 3 |
| 5 | leave_one_sensor_out | 其余 3 sensor 联合 | GelsightMarker | C2 多源 | 1 |
| 6–8 | fewshot_adaptation | 联合 + k-shot GM | GelsightMarker | C3 恢复 | 3 |
| | | | | **小计** | **8** |

> fewshot shots: 1, 5, 10

### A3. Shape Classification (longtail) → target: ViTacTip

> 选择理由：数据量最大 (1599 test)，结果稳定可靠，与 Force target 不同以保证多元性

| # | Protocol | Source | Target | 支撑 | 数量 |
|---|----------|--------|--------|------|------|
| 1 | within_sensor | ViTacTip | ViTacTip | C1 上界 | 1 |
| 2–7 | cross_sensor | 其余 6 sensor 各 1 | ViTacTip | C1 下降 + C2 单源 | 6 |
| 8 | leave_one_sensor_out | 其余 6 sensor 联合 | ViTacTip | C2 多源 | 1 |
| 9–12 | fewshot_adaptation | 联合 + k-shot VTT | ViTacTip | C3 恢复 | 4 |
| | | | | **小计** | **12** |

> fewshot shots: 1, 5, 10, 20

### Part A 小计：12 + 8 + 12 = **32 个实验**

**一张图讲完 C1–C3（以 Force 为例）：**

```
Performance ▲
            │
  within ── │ ████████████████  MagicGripper (C1 上界)
            │
            │ ░░░░             GelsightMarker → MG
            │ ░░░░░░           GelsightNoMarker → MG
  cross ──  │ ░░░              TacTip → MG          (C1 显著下降)
            │ ░░░░░            ViTac → MG
            │ ░░               ViTacTip → MG
            │ ░░░░             ViTacTip2 → MG
            │
  loso ──── │ ▓▓▓▓▓▓▓▓         ALL → MG             (C2 优于多数 cross)
            │
            │ ████████         k=0.005
  fewshot ─ │ ██████████       k=0.01               (C3 逐步恢复)
            │ ████████████     k=0.05
            │ ██████████████   k=0.1
            └──────────────────────────────────────
```

---

## Part B — Representation Study (C4)

**目的**：证明 MAE 自监督预训练能为多个 tactile vision 任务提供统一表征

**设计**：5 个 model config × within_sensor（最简洁的协议，直接衡量表征质量）

| # | Model | Init | 说明 |
|---|-------|------|------|
| 1 | resnet18 | ImageNet | CNN baseline |
| 2 | convnext_tiny | ImageNet | 现代 CNN |
| 3 | vit_small_patch16_224 | random | 无预训练 |
| 4 | vit_small_patch16_224 | ImageNet | 监督预训练 |
| 5 | vit_small_patch16_224 | MAE | **自监督预训练 (本文方法)** |

**每个 model config 只跑 within_sensor，覆盖所有 sensor：**

| 任务 | Sensors | × 5 models | 数量 |
|------|---------|-----------|------|
| Force Regression | 7 sensors | × 5 | 35 |
| Grating (pattern) | 4 sensors | × 5 | 20 |
| Shape (longtail) | 7 sensors | × 5 | 35 |
| **小计** | | | **90** |

**输出的论文表格（Table: Representation Study）：**

```
                     resnet18  convnext  vit_rand  vit_img  vit_mae
                     --------  --------  --------  -------  -------
Force (RMSE↓)
  GelsightMarker       0.29      ...       ...      ...      ...
  GelsightNoMarker     0.29      ...       ...      ...      ...
  MagicGripper         0.27      ...       ...      ...      ...
  TacTip               0.24      ...       ...      ...      ...
  ViTac                0.26      ...       ...      ...      ...
  ViTacTip             0.08      ...       ...      ...      ...
  ViTacTip2            0.06      ...       ...      ...      ...

Grating (Acc↑)
  GelsightMarker       ...       ...       ...      ...      ...
  GelsightNoMarker     1.00      ...       ...      ...      ...
  MagicGripper         ...       ...       ...      ...      ...
  ViTacTip             ...       ...       ...      ...      ...

Shape (Acc↑)
  GelsightMarker       0.98      ...       ...      ...      ...
  GelsigntNoMarker     0.99      ...       ...      ...      ...
  MagicGripper         0.95      ...       ...      ...      ...
  ...                  ...       ...       ...      ...      ...
```

---

## 全局汇总

| Part | 内容 | Backbone | 实验数 |
|------|------|----------|--------|
| A1 | Force C1–C3 | vit/imagenet | 12 |
| A2 | Grating C1–C3 | vit/imagenet | 8 |
| A3 | Shape C1–C3 | vit/imagenet | 12 |
| B | Representation Study | 5 models | 90 |
| **TOTAL** | | | **122** |

### vs 原计划

| | 原计划 | 本方案 | 减少 |
|---|--------|--------|------|
| 总实验数 | 1650 | **122** | **-92.6%** |
| Model configs | 5 (全套) | 1 (主结论) + 5 (rep study) | 分离 |
| Target sensors | 全排列 | 每任务固定 1 个 | -85% |
| Grating variants | 3 | 1 (pattern) | -67% |
| Shape versions | 2 (bal + lt) | 1 (longtail) | -50% |

---

## Target Sensor 选择总结

| 任务 | Target | 技术类型 | 选择理由 |
|------|--------|---------|---------|
| Force | MagicGripper | 磁性传感器 | 中等性能，跨传感器落差清晰 |
| Grating | GelsightMarker | GelSight 光学 | 最经典传感器，论文标杆 |
| Shape | ViTacTip | ViTac 光学 | 数据量最大，结果稳定 |

> 三个 target 分属不同传感器技术，覆盖磁性 / GelSight 光学 / ViTac 光学

---

## 执行参数

### Part A 启动命令

```bash
# Force — ViT/ImageNet, target=MagicGripper
CUDA_VISIBLE_DEVICES=0 python code/force_regression.py \
    --models vit_small_patch16_224 \
    --vit-init-modes imagenet \
    --sensors MagicGripper,GelsightMarker,GelsightNoMarker,TacTip,ViTac,ViTacTip,ViTacTip2 \
    --protocols within,cross,loso,fewshot \
    --image-size 224 --epochs 30 --batch-size 128 \
    --lr 1e-4 --weight-decay 1e-4 --patience 7 --seed 42 \
    --output results/force_minimal.csv

# Grating — ViT/ImageNet, pattern only
CUDA_VISIBLE_DEVICES=1 python code/grating_classification.py \
    --models vit_small_patch16_224 \
    --vit-init-modes imagenet \
    --variants pattern \
    --protocols within,cross,loso,fewshot \
    --image-size 224 --epochs 30 --batch-size 128 \
    --lr 1e-4 --weight-decay 1e-4 --patience 7 --seed 42 \
    --output results/grating_minimal.csv

# Shape — ViT/ImageNet, longtail only
CUDA_VISIBLE_DEVICES=2 python code/shape_classification.py \
    --models vit_small_patch16_224 \
    --vit-init-modes imagenet \
    --variants longtail \
    --protocols within,cross,loso,fewshot \
    --image-size 224 --epochs 30 --batch-size 128 \
    --lr 1e-4 --weight-decay 1e-4 --patience 7 --seed 42 \
    --output results/shape_minimal.csv
```

> ⚠️ 注意：以上命令会跑固定 target 的全部 protocols。如果脚本不支持 `--target-sensor` 参数过滤，
> 输出 CSV 会包含全 sensor 矩阵，但实际需要的数据只有固定 target 的那些行。
> 可以跑完后用 pandas 过滤，也可以修改脚本添加 `--target-sensor` 参数。

### Part B 启动命令

```bash
# Representation Study — 5 models, within_sensor only
for GPU in 0 1 2; do
    TASK=("force_regression" "grating_classification" "shape_classification")
    OUTPUT=("results/rep_force.csv" "results/rep_grating.csv" "results/rep_shape.csv")
    EXTRA=("" "--variants pattern" "--variants longtail")

    CUDA_VISIBLE_DEVICES=$GPU python code/${TASK[$GPU]}.py \
        --models resnet18,convnext_tiny,vit_small_patch16_224 \
        --vit-init-modes random,imagenet,mae \
        --protocols within \
        ${EXTRA[$GPU]} \
        --image-size 224 --epochs 30 --batch-size 128 \
        --lr 1e-4 --weight-decay 1e-4 --patience 7 --seed 42 \
        --output ${OUTPUT[$GPU]} &
done
wait
```

---

## 论文结构映射

| 论文 Section | 数据来源 | 实验数 |
|-------------|---------|--------|
| §5.1 Within-Sensor Baselines | Part B (rep study) | 90 |
| §5.2 Cross-Sensor Transfer Gap | Part A cross_sensor | 15 |
| §5.3 Multi-Source Aggregation | Part A loso vs cross | (复用上面) |
| §5.4 Few-Shot Adaptation | Part A fewshot | 11 |
| §5.5 Representation Study | Part B | (复用 §5.1) |
| **论文需要的独立实验总数** | | **122** |
