# 与原始代码的关键修改说明

---

# 1. 奖励函数重设计（Reward Shaping）

## 1.1 新增非对称步态惩罚（`asymmetry_penalty`）

**位置**：`walker_reward()` 函数

### 修改内容

在原有 `leg_balance_reward` 基础上，新增基于两条腿绝对活动度差异的对称性惩罚项。

### 数学形式

```python
asymmetry_penalty = tolerance(
    abs(right_activity - left_activity) / (right_activity + left_activity + 1e-6),
    bounds=(0.0, 0.5),
    margin=0.25,
    value_at_margin=0.2,
    sigmoid="linear",
)
```

### 动机

原始奖励仅通过 `leg_activity_ratio`（最小值 / 最大值）约束两条腿的比例关系，无法有效惩罚：

- 一条腿极度活跃
- 另一条腿仅轻微摆动

这种“拖曳步态（dragging gait）”。

引入绝对差异惩罚后，两条腿的活动度必须保持相近，否则 `gait_reward` 会被直接缩放，从而强制策略学习对称的双足发力模式。

---

## 1.2 交替步态权重内联与保底降低

**位置**：`walker_reward()` 函数中的 `gait_reward` 计算部分

### 修改内容

将原来的：

```python
alternation_factor = 1.0 - 0.8 + 0.8 * alternation_reward
```

改为直接内联：

```python
(0.1 + 0.9 * alternation_reward)
```

即：

- 原始保底：20%
- 当前保底：10%

### 动机

原始公式即使完全不交替，也仍保留：

```python
20% gait_reward
```

这给“拖腿行走”留下了奖励漏洞。

降低保底后：

- 策略必须真正完成左右脚交替
- 才能获得可观的步态奖励

从而显著抑制非自然步态。

---

## 1.3 速度奖励保底移除

**位置**：`walker_reward()` 函数最终 `return` 语句

### 修改内容

原始代码：

```python
return float(posture_reward * gait_reward * (0.2 + 0.8 * move_reward))
```

当前代码：

```python
return float(posture_reward * gait_reward * move_reward)
```

### 动机

原始公式中：

即使前进速度为 0，也能保留：

```python
0.2 * posture * gait
```

的基础奖励。

这导致策略可以通过：

- 原地摆动
- 假装迈步
- 不实际前进

来获取正奖励。

移除保底后：

- 必须产生真实前进速度
- 才能获得正回报

从而彻底消除“静止作弊”。

---

## 1.4 `gait_reward` 综合公式重构

### 原始公式

```python
gait_reward = (
    leg_balance_reward
    * foot_separation_reward
    * alternation_factor
)
```

### 当前公式

```python
gait_reward = (
    leg_balance_reward
    * foot_separation_reward
    * (0.1 + 0.9 * alternation_reward)
    * asymmetry_penalty
)
```

### 修改效果

对于正常交替步态：

- 所有惩罚项 ≈ 1.0
- 可获得完整奖励

对于：

- 拖腿
- 单腿主导
- 非对称步态

则会被多个惩罚项联乘压制。

最终效果是：

TD3 优化器会自然偏向于学习稳定、对称、交替的双足步态，而不是利用奖励漏洞的局部最优策略。

---

# 2. 摔倒检测与早期终止（Fall Detection & Early Termination）

---

## 2.1 物理摔倒检测函数

### 位置

新增 `check_termination()` 函数

### 代码

```python
def check_termination(physics):
    torso_height = physics_value(physics.named.data.xpos["torso", "z"])
    return torso_height < 0.6
```

### 动机

原始代码没有摔倒检测机制。

因此机器人即使已经倒地：

- 环境仍继续运行满 1000 步
- 持续采样无效状态
- Replay Buffer 被大量污染

新增基于躯干高度的检测后：

- 当机器人真正摔倒时
- 环境可立即终止 episode

同时：

仅使用高度判断，而不使用倾角判断，可以避免：

- 加速姿态
- 前倾奔跑
- 动态摆动

被误判为摔倒。

---

## 2.2 训练循环终止逻辑重构

### 位置

`train()` 主循环

### 核心修改

引入统一的 episode 结束标志：

```python
episode_over = bool(next_ts.last())
```

并增加摔倒检测：

```python
if args.domain == "walker" and check_termination(env.physics):
    episode_over = True
    reward -= 3.0
```

随后统一使用：

```python
if episode_over:
```

替代原始：

```python
if ts.last():
```

### 动机

#### （1）避免重复惩罚

原始逻辑中：

如果在循环内部直接修改 reward：

- 摔倒后每一步都会继续扣分
- 躺地 1000 步可能累计：
  
```python
-3000 reward
```

当前逻辑中：

- 仅在摔倒瞬间惩罚一次
- 随后立即 reset 环境

更加符合真实 RL termination 逻辑。

---

#### （2）避免 Replay Buffer 污染

原始代码存在严重问题：

- 人为写入 `done = 1`
- 但环境实际上并未 reset

导致 Replay Buffer 中存入：

```python
(s, a, r, s', done=1)
```

但：

```python
s'
```

实际上仍是：

- 倒地后挣扎状态
- 无效物理状态
- 非真正 terminal state

这会严重破坏 TD3 的 bootstrap 目标估计。

当前逻辑中：

- `episode_over=True`
- 立即 reset_env()
- done 与真实终止严格一致

从而保证：

- Bellman target 正确
- Replay Buffer 数据干净
- critic 学习稳定

---

# 3. 跨平台渲染兼容性

### 位置

`parse_args()` 中：

```python
--render_backend
```

默认值修改。

### 修改内容

原始：

```python
default="egl"
```

当前：

```python
default="glfw"
```

### 动机

原始默认 `egl` 为 Linux 无头渲染后端。

在 Windows 下运行时会报错：

```python
ImportError: Unable to load EGL library
```

改为 `glfw` 后：

- Windows 可直接运行
- 无需额外参数
- 支持本地视频渲染与训练

提升了跨平台兼容性。

---

# 4. 代码结构优化

---

## 4.1 `done` 语义优化

### 修改内容

原始：

```python
done = float(next_ts.last())
```

当前：

```python
episode_over = bool(next_ts.last())
```

### 优势

相比 `done`：

- `episode_over`
  语义更加明确
- 更符合 RL 中：
  
```text
episode termination
```

的真实含义。

---

## 4.2 Replay Buffer 写入统一化

### 修改内容

统一使用：

```python
float(episode_over)
```

写入 Buffer。

### 优势

保证：

- 逻辑判断
- buffer done 标记
- 环境 reset 时机

三者完全同步。

避免：

- hidden state mismatch
- bootstrap 错误
- termination 不一致

---

## 4.3 显式环境推进逻辑

### 修改内容

使用：

```python
if episode_over:
    ...
else:
    obs = next_obs
    ts = next_ts
```

替代隐式推进。

### 优势

避免：

- episode 已结束
- 却仍继续推进环境状态

导致的隐藏 bug。

同时：

代码可读性与维护性也显著提高。

---

# 修改效果总结

| 指标 | 原始代码 | 当前代码 |
|---|---|---|
| 拖腿步态 | 可获取高回报（约 130） | 被 `asymmetry_penalty` 与低 alternation 保底压制 |
| 静止作弊 | 速度保底 0.2，原地摆动可拿分 | 速度保底移除，必须真实前进 |
| 摔倒后行为 | 无检测，躺地运行满 1000 步 | 立即检测、一次性惩罚、立即 reset |
| Replay Buffer 质量 | done 与 reset 不同步 | done 与 reset 严格一致 |
| TD3 Bootstrap 稳定性 | 存在伪终止状态污染 | terminal state 真实可靠 |
| 跨平台兼容性 | Windows 需手动指定后端 | 默认 `glfw` 开箱即用 |
| 代码可维护性 | 终止逻辑分散 | episode_over 统一管理 |

---

# 总体效果

这些修改共同作用，使策略优化目标从：

- 利用奖励漏洞
- 原地震荡
- 拖腿移动
- 单腿作弊

转向：

- 对称双足发力
- 稳定交替步态
- 持续真实前进
- 更符合自然行走规律的 locomotion policy

最终显著提升：

- 步态自然性
- 训练稳定性
- Replay Buffer 数据质量
- TD3 收敛效果
- 跨平台可运行性
```
