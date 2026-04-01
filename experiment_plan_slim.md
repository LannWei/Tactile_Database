# TacVerse Benchmark — 精简实验计划

> 目标：从 5 个 model config 精简到 **2 个**，减少 60% 实验量
> 原则：最大化多元性 — CNN vs Transformer, 监督 vs 自监督

---

## 1. 保留的 2 个 Model Config

| # | Backbone | Init | 选择理由 |
|---|----------|------|---------|
| 1 | `resnet18` | ImageNet | CNN baseline，训练快，经典对照组 |
| 2 | `vit_small_patch16_224` | MAE | ViT + 自监督预训练，本项目核心贡献点 |

**淘汰的 3 个：**
- ~~`convnext_tiny / imagenet`~~ — 与 resnet18 同属 CNN 系列，冗余
- ~~`vit / random`~~ — 无预训练的 ViT 通常表现差，参考价值低
- ~~`vit / imagenet`~~ — 与 MAE 相比，监督预训练不是本文重点

> **对比意义**：resnet18 (CNN + 监督) vs ViT-MAE (Transformer + 自监督) = 架构差异 × 预训练策略差异，一张表就能说明问题。

---

## 2. 精简后实验量

### 2.1 Force Regression (GPU 0)

| Protocol | 数量 |
|----------|------|
| within_sensor | 7 |
| cross_sensor | 42 |
| leave_one_sensor_out | 7 |
| fewshot_adaptation | 28 |
| **小计 (per config)** | **84** |
| **× 2 configs** | **168** |

### 2.2 Grating Classification (GPU 1)

| Protocol | 数量 |
|----------|------|
| within_sensor | 4 |
| cross_sensor | 12 |
| leave_one_sensor_out | 4 |
| fewshot_adaptation | 12 |
| **小计 (per variant per config)** | **32** |
| **× 3 variants × 2 configs** | **192** |

### 2.3 Shape Balanced (GPU 2)

| Protocol | 数量 |
|----------|------|
| within_sensor | 6 |
| cross_sensor | 30 |
| leave_one_sensor_out | 6 |
| fewshot_adaptation | 24 |
| **小计 (per config)** | **66** |
| **× 2 configs** | **132** |

### 2.4 Shape Longtail (GPU 3)

| Protocol | 数量 |
|----------|------|
| within_sensor | 7 |
| cross_sensor | 42 |
| leave_one_sensor_out | 7 |
| fewshot_adaptation | 28 |
| **小计 (per config)** | **84** |
| **× 2 configs** | **168** |

---

## 3. 前后对比

| GPU | 实验 | 原 (5 configs) | 新 (2 configs) | 减少 |
|-----|------|---------------|----------------|------|
| 0 | Force Regression | 420 | **168** | -252 |
| 1 | Grating Classification | 480 | **192** | -288 |
| 2 | Shape Balanced | 330 | **132** | -198 |
| 3 | Shape Longtail | 420 | **168** | -252 |
| | **TOTAL** | **1650** | **660** | **-990 (60%)** |

---

## 4. 已有数据可复用

resnet18 已经在跑，当前进度：

| GPU | 实验 | resnet18 已完成 | resnet18 总需 | resnet18 剩余 |
|-----|------|----------------|--------------|--------------|
| 0 | Force | 50 | 84 | 34 |
| 1 | Grating | 33 | 96 | 63 |
| 2 | Shape Balanced | 58 | 66 | 8 |
| 3 | Shape Longtail | 59 | 84 | 25 |

所以实际还需新跑的：

| 阶段 | 数量 | 说明 |
|------|------|------|
| resnet18 剩余 | 130 | 当前正在跑，等它跑完 |
| vit/mae 全量 | 330 | 新启动 (168+96+66+84... 不对) |

**修正：vit/mae 全量**
- Force: 84
- Grating: 32 × 3 = 96
- Shape Balanced: 66
- Shape Longtail: 84
- 合计: **330**

**实际还需跑：130 (resnet18 剩余) + 330 (vit/mae) = 460 个实验**
vs 原计划剩余 1450 个，**减少 68%**。

---

## 5. 执行方案

### 方案 A：修改 `run_benchmark.sh`（推荐）

在启动命令中添加 `--models` 参数限制 backbone：

```bash
# 公共参数增加 model 限制
COMMON=(
    --models resnet18,vit_small_patch16_224
    --vit-init-modes mae
    --image-size 224
    --epochs 30
    --batch-size 128
    --lr 1e-4
    --weight-decay 1e-4
    --patience 7
    --num-workers 4
    --seed 42
    --mae-epochs 100
    --mae-batch-size 256
    --mae-lr 1.5e-4
    --mae-weight-decay 0.05
    --mae-mask-ratio 0.75
)
```

关键变更：
- `--models resnet18,vit_small_patch16_224` — 只跑这 2 个 backbone
- `--vit-init-modes mae` — ViT 只跑 MAE 初始化

### 方案 B：先停掉当前进程，改参数重跑

```bash
# 1. 停掉当前 benchmark
kill <benchmark_pid>

# 2. 用新参数重跑（已完成的 resnet18 结果会自动跳过或覆盖）
nohup bash run_benchmark.sh &
```

---

## 6. 论文覆盖度检查

精简后仍可支持的论文内容：

| 论文表格/图 | 需要的数据 | ✅/❌ |
|-------------|----------|------|
| Within-sensor 性能表 | 所有 sensor × 2 models | ✅ |
| Cross-sensor 泛化矩阵 | N×N sensor 对 × 2 models | ✅ |
| LOSO 泛化表 | N-1 train → 1 test × 2 models | ✅ |
| Few-shot 曲线 | shots vs accuracy × 2 models | ✅ |
| CNN vs ViT 对比 | resnet18 vs vit_mae | ✅ |
| MAE 预训练增益 | vit_mae 直接看 | ✅ |
| 多 backbone 全面对比 | 需要 5 个 model | ❌ 仅 2 个 |
| ConvNeXt 性能 | 需要 convnext | ❌ 已删除 |

> 如果审稿人要求更多 backbone 对比，后续可补跑 `convnext_tiny` 或 `vit/imagenet`。
