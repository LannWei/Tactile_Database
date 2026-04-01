# TacVerse Benchmark 实验总览

> 运行环境: 4× H200 (96GB)
> 启动时间: Mon Mar 23 12:22:24 UTC 2026
> 快照时间: Mon Mar 24 10:33 UTC 2026

---

## 1. Model × Init 配置 (所有实验共用)

| # | Backbone | Init | 说明 |
|---|----------|------|------|
| 1 | `resnet18` | ImageNet | CNN baseline |
| 2 | `convnext_tiny` | ImageNet | 现代 CNN |
| 3 | `vit_small_patch16_224` | Random | ViT 随机初始化 |
| 4 | `vit_small_patch16_224` | ImageNet | ViT ImageNet 预训练 |
| 5 | `vit_small_patch16_224` | MAE | ViT 自监督 MAE 预训练 |

共 **5** 种 model config，每个实验的每组 protocol 都要跑完这 5 种。

---

## 2. 四大实验

### 2.1 Force Regression (GPU 0)

| 属性 | 值 |
|------|------|
| 脚本 | `code/force_regression.py` |
| 任务 | 力回归 (RMSE / MAE) |
| Sensors (7) | GelsightMarker, GelsightNoMarker, MagicGripper, TacTip, ViTac, ViTacTip, ViTacTip2 |
| Fewshot ratios | 0.005, 0.01, 0.05, 0.1 (4 级) |

**每 model config 实验数：**

| Protocol | 计算方式 | 数量 |
|----------|---------|------|
| within_sensor | 7 sensors 各自 train/test | **7** |
| cross_sensor | 7 × 6 (所有 source→target 对) | **42** |
| leave_one_sensor_out | 6 train → 1 test × 7 targets | **7** |
| fewshot_adaptation | 7 targets × 4 ratios | **28** |
| **小计** | | **84** |

**总计：84 × 5 configs = 420**

| 状态 | 进度 |
|------|------|
| ✅ resnet18 within_sensor | 7/7 |
| ✅ resnet18 cross_sensor | 42/42 |
| 🔄 resnet18 leave_one_sensor_out | 1/7 |
| ⬜ resnet18 fewshot_adaptation | 0/28 |
| ⬜ convnext_tiny | 0/84 |
| ⬜ vit / random | 0/84 |
| ⬜ vit / imagenet | 0/84 |
| ⬜ vit / mae | 0/84 |
| **合计** | **50 / 420 (11.9%)** |

---

### 2.2 Grating Classification (GPU 1)

| 属性 | 值 |
|------|------|
| 脚本 | `code/grating_classification.py` |
| 任务 | 光栅纹理分类 (Accuracy / F1) |
| Sensors (4) | GelsightMarker, GelsightNoMarker, MagicGripper, ViTacTip |
| Variants | pattern, size, joint (3 种分类任务) |
| Fewshot shots | 1, 5, 10 (3 级) |

**每 variant 每 model config 实验数：**

| Protocol | 计算方式 | 数量 |
|----------|---------|------|
| within_sensor | 4 sensors | **4** |
| cross_sensor | 4 × 3 | **12** |
| leave_one_sensor_out | 3 train → 1 test × 4 | **4** |
| fewshot_adaptation | 4 targets × 3 shots | **12** |
| **小计** | | **32** |

**总计：32 × 3 variants × 5 configs = 480**

| 状态 | 进度 |
|------|------|
| ✅ pattern / resnet18 | 32/32 |
| 🔄 pattern / convnext_tiny | 1/32 |
| ⬜ pattern / vit×3 | 0/96 |
| ⬜ size (全部) | 0/160 |
| ⬜ joint (全部) | 0/160 |
| **合计** | **33 / 480 (6.9%)** |

---

### 2.3 Shape Classification — Balanced (GPU 2)

| 属性 | 值 |
|------|------|
| 脚本 | `code/shape_classification.py --variants balanced` |
| 任务 | 形状分类，balanced 数据集 (Accuracy / F1) |
| Sensors (6) | GelsigntNoMarker, MagicGripper, MagicTac, TacTip, ViTac, ViTacTip |
| 说明 | 仅使用各类样本数均为 500 的 6 个 balanced sensors |
| Fewshot shots | 1, 5, 10, 20 (4 级) |

**每 model config 实验数：**

| Protocol | 计算方式 | 数量 |
|----------|---------|------|
| within_sensor | 6 sensors | **6** |
| cross_sensor | 6 × 5 | **30** |
| leave_one_sensor_out | 5 train → 1 test × 6 | **6** |
| fewshot_adaptation | 6 targets × 4 shots | **24** |
| **小计** | | **66** |

**总计：66 × 5 configs = 330**

| 状态 | 进度 |
|------|------|
| ✅ resnet18 within_sensor | 6/6 |
| ✅ resnet18 cross_sensor | 30/30 |
| ✅ resnet18 loso | 6/6 |
| 🔄 resnet18 fewshot | 16/24 |
| ⬜ convnext_tiny | 0/66 |
| ⬜ vit / random | 0/66 |
| ⬜ vit / imagenet | 0/66 |
| ⬜ vit / mae | 0/66 |
| **合计** | **58 / 330 (17.6%)** |

---

### 2.4 Shape Classification — Longtail (GPU 3)

| 属性 | 值 |
|------|------|
| 脚本 | `code/shape_classification.py --variants longtail` |
| 任务 | 形状分类，longtail 数据集 (Accuracy / F1) |
| Sensors (7) | GelsightMarker, GelsigntNoMarker, MagicGripper, MagicTac, TacTip, ViTac, ViTacTip |
| 说明 | 使用全部 7 个 sensor（含不均衡的 GelsightMarker），class_weight=balanced |
| Fewshot shots | 1, 5, 10, 20 (4 级) |

**每 model config 实验数：**

| Protocol | 计算方式 | 数量 |
|----------|---------|------|
| within_sensor | 7 sensors | **7** |
| cross_sensor | 7 × 6 | **42** |
| leave_one_sensor_out | 6 train → 1 test × 7 | **7** |
| fewshot_adaptation | 7 targets × 4 shots | **28** |
| **小计** | | **84** |

**总计：84 × 5 configs = 420**

| 状态 | 进度 |
|------|------|
| ✅ resnet18 within_sensor | 7/7 |
| ✅ resnet18 cross_sensor | 42/42 |
| ✅ resnet18 loso | 7/7 |
| 🔄 resnet18 fewshot | 3/28 |
| ⬜ convnext_tiny | 0/84 |
| ⬜ vit / random | 0/84 |
| ⬜ vit / imagenet | 0/84 |
| ⬜ vit / mae | 0/84 |
| **合计** | **59 / 420 (14.0%)** |

---

## 3. 全局汇总

| GPU | 实验 | Per Config | Configs | Variants | 总计 | 已完成 | 剩余 | 进度 |
|-----|------|-----------|---------|----------|------|--------|------|------|
| 0 | Force Regression | 84 | 5 | 1 | **420** | 50 | 370 | 11.9% |
| 1 | Grating Classification | 32 | 5 | 3 | **480** | 33 | 447 | 6.9% |
| 2 | Shape Balanced | 66 | 5 | 1 | **330** | 58 | 272 | 17.6% |
| 3 | Shape Longtail | 84 | 5 | 1 | **420** | 59 | 361 | 14.0% |
| | **TOTAL** | | | | **1650** | **200** | **1450** | **12.1%** |

---

## 4. 实验数量来源分析

```
1650 = Force(420) + Grating(480) + ShapeBal(330) + ShapeLT(420)

其中:
  Force    = (7w + 42c + 7l + 28f) × 5 = 84 × 5 = 420
  Grating  = (4w + 12c + 4l + 12f) × 3v × 5 = 32 × 15 = 480
  ShapeBal = (6w + 30c + 6l + 24f) × 5 = 66 × 5 = 330
  ShapeLT  = (7w + 42c + 7l + 28f) × 5 = 84 × 5 = 420

w = within_sensor, c = cross_sensor, l = loso, f = fewshot
v = variant, × 5 = 5 model configs
```

最大贡献者：
- **cross_sensor O(N²)** — 7 sensors 产生 42 对，占单 config 实验的 50%
- **fewshot × shot levels** — 每个 target sensor × 每个 shot 数
- **5 个 model configs** — 论文需对比 ResNet/ConvNeXt/ViT×3
- **Grating 3 个 variant** — pattern/size/joint 各跑全套

---

## 5. Protocol 说明

| Protocol | 含义 | 目的 |
|----------|------|------|
| `within_sensor` | 同一 sensor 的数据训练和测试 | 衡量单传感器上限 |
| `cross_sensor` | sensor A 训练 → sensor B 测试 | 衡量跨传感器泛化 |
| `leave_one_sensor_out` | N-1 个 sensor 训练 → 剩余 1 个测试 | 衡量多传感器聚合的泛化 |
| `fewshot_adaptation` | N-1 个 sensor 预训练 + 少量 target 数据微调 → target 测试 | 衡量少样本迁移能力 |

---

## 6. 运行配置

```bash
# 公共超参
--image-size 224
--epochs 30
--batch-size 128
--lr 1e-4
--weight-decay 1e-4
--patience 7        # early stopping
--num-workers 4
--seed 42

# MAE 预训练 (Phase 1, 已完成)
--mae-epochs 100
--mae-batch-size 256
--mae-lr 1.5e-4
--mae-weight-decay 0.05
--mae-mask-ratio 0.75
```

---

## 7. 输出文件

| 文件 | 说明 |
|------|------|
| `results/force_regression_results.csv` | Force 实验全部结果 |
| `results/grating_classification_results.csv` | Grating 实验全部结果 |
| `results/shape_classification_balanced.csv` | Shape balanced 结果 |
| `results/shape_classification_longtail.csv` | Shape longtail 结果 |
| `results/shape_classification_results.csv` | Shape 合并结果 (Phase 3) |
| `results/paper_tables/` | 论文表格 (Phase 4) |
| `results/paper_figures/` | 论文图表 (Phase 4) |
| `results/logs/gpu{0-3}_*.log` | 各 GPU 日志 |
