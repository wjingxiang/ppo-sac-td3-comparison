# PPO / SAC / TD3 深度强化学习算法对比

基于 [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) 实现的三个深度强化学习算法（PPO、SAC、TD3）在连续控制任务上的系统对比，包含 4 项指标评测与 Optuna 贝叶斯超参数调优。

> 课程项目：COMP 4330SEF · P04 Group 2

## 结果摘要（LunarLanderContinuous-v3，5 个种子平均）

| 指标 | PPO | SAC | TD3 |
|---|---|---|---|
| 最终回报 | 141.6 ± 32.6 | 223.9 ± 83.5 | **284.1 ± 3.0** |
| 采样效率（50% 预算） | 129.1 | 224.8 | **277.0** |
| 训练稳定性（标准差，越小越稳） | 68.1 | 93.5 | **30.8** |
| 收敛率（达 200 分） | 60% | 80% | **100%** |

**结论**：离策略算法（SAC / TD3）明显优于在策略的 PPO；其中 TD3 依靠「双 Critic + 延迟更新」抑制了价值高估，在四个指标上全面胜出。

## 环境要求

- Python 3.11（Windows / Linux / macOS 均可）
- 建议 16GB 内存（SAC/TD3 的经验回放池占内存较大）
- **无需 GPU**（本任务网络很小，瓶颈在 CPU 物理模拟，GPU 几乎无收益）

## 安装

```bash
# 1. 创建虚拟环境
py -m venv .venv

# 2. 激活（Windows）
.venv\Scripts\activate
# 或 Linux / macOS
source .venv/bin/activate

# 3. 安装依赖
py -m pip install -U pip
py -m pip install -r requirements.txt
```

`requirements.txt` 内容：`stable-baselines3`、`gymnasium[box2d]`、`mujoco`、`optuna`、`matplotlib`、`pandas`、`imageio`。

> 注意：新版 gymnasium（≥1.4）已移除 `LunarLanderContinuous-v2`，本项目使用 `-v3`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `config.py` | 全局配置：环境、训练步数、随机种子、收敛阈值（改参数只改这里） |
| `experiment.py` | 单任务训练 + 周期评估 + 最终验证（支持断点续跑） |
| `run_baseline.py` | 顺序跑完所有基线组合 |
| `run_parallel.py` | 并行跑基线（`--workers N` 多核加速） |
| `compute_metrics.py` | 计算 4 项指标 |
| `plot_curves.py` | 画学习曲线 |
| `tune.py` | Optuna 贝叶斯调参（SQLite 落盘 + 断点续跑） |
| `validate_tuned.py` | 用最优参数全预算重训验证（单种子） |
| `run_validation.py` | 并行启动 5 个种子的重训验证 |

## 使用方法

### 1. 冒烟测试（先验证环境装好）

```bash
py experiment.py --algo TD3 --env LunarLanderContinuous-v3 --seed 0 --smoke
```

### 2. 跑完整基线

```bash
# 单个任务（3 算法 × 2 环境 × 5 种子）
py experiment.py --algo PPO --env LunarLanderContinuous-v3 --seed 0

# 并行跑（推荐，多核 CPU 提速）
py run_parallel.py --env LunarLanderContinuous-v3 --workers 4 --threads 1
```

### 3. 算指标 + 画图

```bash
py compute_metrics.py    # 打印 4 项指标，存 logs/<env>/summary.csv
py plot_curves.py        # 存 logs/<env>/learning_curves.png
```

### 4. 超参数调优（Optuna）

```bash
py tune.py --algo TD3 --env LunarLanderContinuous-v3 --trials 20
# 最优参数存 logs/<env>/TD3_tune_best.json
# 冒烟测试可用 --steps 2000 缩水预算
```

### 5. 用最优参数全预算重训验证

```bash
py run_validation.py     # 并行跑 5 个种子，结果存 TD3_tuned_seed*.final.csv
```

## 注意事项

- **内存**：SAC/TD3 的经验回放池（1M 条）占内存较大；并行跑时 `--workers` 建议 ≤4，并关闭其他占内存的程序。
- **速度**：SAC/TD3 是离策略算法，每步做一次梯度更新，比 PPO 慢（单进程约 110 步/秒）是正常的。
- **收敛阈值**：`config.py` 里的 `threshold` 需按环境设定（LunarLander 用 200 分「解决」标准线）。
- **断点续跑**：`experiment.py` 和 `tune.py` 支持断点续跑，中途被杀可自动从断点恢复，不会从头再来。
