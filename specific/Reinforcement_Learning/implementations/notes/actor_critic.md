# Actor-Critic (AC)

## 核心思想

Actor-Critic 结合了 Policy Gradient（actor）和 Value Function（critic）两个组件：

- **Critic**：学习状态价值函数 `V(s)`，作为 baseline
- **Actor**：基于 critic 提供的 advantage 更新策略

## Critic

### 输入/输出

- 输入：状态 `s_t`（不包含动作）
- 输出：`V(s_t)`，即从状态 `s_t` 出发的预期累计 reward

### 学习目标

最小化 Bellman 残差：

```
L_critic = (r_t + γ·V(s_{t+1}) - V(s_t))²
```

目标是让 `V(s_t)` 收敛到 `r_t + γ·V(s_{t+1})`。

## Advantage

```
A(s_t, a_t) = r_t + γ·V(s_{t+1}) - V(s_t)
```

- `A > 0`：这个动作比平均水平好
- `A < 0`：这个动作比平均水平差
- 用的是**更新前**的 `V`（`torch.no_grad()`），不是 critic 更新后的残差

### Advantage vs Critic Residual（常见混淆点）

两者表达式相同，但目的不同：

| | Critic Residual | Advantage |
|--|--|--|
| **用途** | critic 自我更新的误差信号 | actor 判断动作好坏 |
| **使用哪个 V** | 更新后的 V（循环内） | 更新前的 V（`no_grad` 快照） |
| **如果用错** | 没问题 | actor 拿到近零残差，梯度消失 |

**正确顺序**：先用 `no_grad` 计算 advantage → 再循环更新 critic。

## Actor

### 更新目标

```
loss = -log π(a|s) · A(s, a)
```

Actor 训练用的是 **advantage**，不是原始 reward。
Reward 只是原材料，advantage 才是真正的学习信号。

## GAE（Generalized Advantage Estimation）

标准 TD error 是单步估计，GAE 用 λ 加权多步：

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = δ_t + (γλ)·(1 - done_t)·A_{t+1}   # 从后往前递推
```

- `λ=0`：退化为单步 TD（低方差，高偏差）
- `λ=1`：退化为 Monte Carlo（高方差，低偏差）
- 典型值：`λ=0.95`

## 一步流程

```
1. actor 按 π(a|s) 采样动作 a_t
2. 环境返回 r_t, s_{t+1}, done
3. critic 计算 V(s_t), V(s_{t+1})（no_grad）
4. 算 advantage = r_t + γ·V(s_{t+1}) - V(s_t)
5. actor 用 advantage 更新策略
6. critic 用 Bellman 残差更新 V（可多步 v_loss_iter）
```

## 超参参考（CartPole-v1）

```python
hypers = {
    "GAMMA": 0.99,
    "critic_LR": 1e-3,
    "LR": 3e-4,
    "BATCH": 1000,
    "v_loss_iter": 30,
    "GAE": True,
    "TD_lambda": 0.95,
    "update_per_episode": False,
}
```
