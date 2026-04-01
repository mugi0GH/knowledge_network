# PPO (Proximal Policy Optimization)

## 核心思想

PPO 是 TRPO 的简化版，目标相同（限制每次策略更新幅度），但用 clip 替代显式 KL 约束，避免了共轭梯度和 line search 的计算开销。

**PPO = AC + 重要性采样 + Clip**

---

## 与 TRPO 的对比

| | TRPO | PPO |
|--|--|--|
| **约束方式** | 显式 KL 约束 + line search | clip 截断概率比 |
| **计算成本** | 高（CG + line search） | 低（普通梯度下降） |
| **理论保证** | 严格单调改进 | 近似保证，实践效果好 |
| **数据复用** | 每 batch 只更新一次 | 同一 batch 可更新 K epoch |
| **方向性** | 自然梯度 F⁻¹g | 普通梯度（无方向修正） |

PPO **不使用**自然梯度，放弃了 TRPO 的方向矫正，靠 clip 来近似约束。

---

## 重要性采样

用 `π_old` 采集的数据估计 `π_new` 的梯度，需要修正分布偏差：

```
L_IS = mean( r_t · A_t )

r_t = π_new(a_t|s_t) / π_old(a_t|s_t)
    = exp(log_π_new - log_π_old)
```

- `log_π_old`：采集数据时记录，固定不变
- `log_π_new`：每次更新时用当前策略重新计算
- 如果 `r_t = 1`（新旧策略完全一致），退化为普通 PG

---

## Clip Loss（PPO-Clip，主流）

```python
ratio = torch.exp(log_prob_new - log_prob_old)   # r_t，shape: [batch]

surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage

actor_loss = -torch.min(surr1, surr2).mean()
```

典型 `eps = 0.1 ~ 0.2`。

### Clip 的截断逻辑（不对称）

| 情况 | A > 0（好动作）| A < 0（坏动作）|
|--|--|--|
| ratio > 1+ε | **梯度=0**，不再增大概率 | 梯度正常，继续压低概率 |
| ratio < 1-ε | 梯度正常，继续增大概率 | **梯度=0**，不再压低概率 |

规律：**只有"已经更新过度"的方向才截断，"还没更新到位"的方向继续更新。**

触边界不是"取消更新"，是 `clamp` 输出恒为边界值，对 ratio 导数为 0，等效于该样本在这个方向停止更新，但不影响其他样本的梯度。

### "更新够了"的判断依据是 ratio，不是 advantage

- `ratio > 1`：新策略比旧策略更倾向于选这个动作（概率被推高了）
- `ratio < 1`：新策略比旧策略更不倾向于选这个动作（概率被压低了）
- `ratio = 1`：还没动过

Advantage 决定**方向**（该推高还是压低），ratio 衡量**已经走了多远**，`1 ± ε` 是"走够了"的边界。

### 为什么必须先 ratio × advantage 再做 min

如果只 clip ratio 再乘 advantage：

```python
# 错误直觉
loss = -torch.clamp(ratio, 1-eps, 1+eps) * advantage
```

这会两个方向都截断——`A > 0` 时 ratio 低于下界也截断，阻止"还没推够"的情况继续更新，完全错误。

`min(surr1, surr2)` 的截断方向**由 A 的符号自动决定**：

```
A > 0，ratio > 1+ε（推得太多）：
    surr1 = 1.5·A = 1.5  （大）
    surr2 = 1.2·A = 1.2  （小）
    min 选 surr2 → clamp 侧，梯度=0，停止推高 ✓

A > 0，ratio < 1（还没推够）：
    surr1 = 0.8·A = 0.8
    surr2 = 0.8·A = 0.8  （在范围内，clamp 不生效）
    min 选 surr1 → 梯度正常，继续推高 ✓

A < 0，ratio < 1-ε（压得太多）：
    surr1 = 0.5·(-1) = -0.5
    surr2 = 0.8·(-1) = -0.8  （更小）
    min 选 surr2 → clamp 侧，梯度=0，停止压低 ✓
```

`min` 在负数里选更小的，正好选到了 clamp 那侧。**`min` 自动感知 A 的符号，顺序反了这个机制就不成立。**

---

## Advantage（A_t）

**Critic 不直接评估动作**，只评估状态价值 V(s)。Advantage 是在 V(s) 基础上算出来的：

```
A_t = r_t + γ·V(s_{t+1}) - V(s_t)
      ↑ 真实 reward    ↑ critic 估计的基准
```

### GAE 版本（推荐）

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ·(1-done_t)·A_{t+1}    # 从后往前递推
```

GAE 是 bias-variance 的权衡：
- `λ=0`：单步 TD，低方差高偏差
- `λ=1`：Monte Carlo，低偏差高方差
- `λ=0.95`：典型值，中间地带

**GAE 不一定比普通 Advantage 好**，前提是 V 学得不太差也不太准。Episode 长短差异大时 GAE 优势明显。

---

## 训练流程

```python
# 1. 用 π_old 采集 rollout
for t in range(rollout_steps):
    action, log_prob_old = actor.sample(state)
    next_state, reward, done = env.step(action)
    buffer.store(state, action, reward, done, log_prob_old)

# 2. 算 GAE advantage（固定，整个训练期间不变）
advantages = compute_gae(buffer, critic)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# 3. K epoch 更新
for epoch in range(K):
    indices = torch.randperm(N)          # 每 epoch 重新 shuffle
    for i in range(0, N, minibatch_size):
        mb = indices[i:i+minibatch_size]

        log_prob_new = actor.log_prob(states[mb], actions[mb])
        ratio = torch.exp(log_prob_new - log_prob_old[mb])

        surr1 = ratio * advantages[mb]
        surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages[mb]
        actor_loss = -torch.min(surr1, surr2).mean()

        value_loss = (critic(states[mb]) - returns[mb]).pow(2).mean()

        loss = actor_loss + 0.5 * value_loss - entropy_coef * entropy
        loss.backward()
        optimizer.step()

# 4. π_old ← π_new，开始下一轮 rollout
```

### Shuffle 的作用

- 同一 rollout 内部打乱，不跨 rollout
- 消除时序相关性：相邻时间步状态高度相似，按顺序切 minibatch 会导致梯度估计偏斜
- 随机子集期望分布等于总体分布（统计基本结论）

---

## 实现细节与常见坑

### 1. log_prob_old 必须在采集时记录

不能事后用旧网络重算，因为 `π_old` 在 rollout 结束后就会被更新覆盖。

```python
# 采集时
log_prob_old = dist.log_prob(action).detach()   # detach 很重要
```

### 2. Advantage 归一化时机

在整个 rollout 算完 GAE 之后、进入 epoch 循环之前归一化一次，不要在每个 minibatch 里重新归一化。

### 3. Critic 的 target

用 GAE returns（advantage + V(s)），不要只用 reward-to-go：

```python
returns = advantages + values   # GAE returns 作为 critic target
```

### 4. Entropy bonus

加 entropy 正则防止策略过早收敛：

```python
loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
```

### 5. Early stopping（可选）

如果 epoch 内平均 KL 超过软阈值（通常 1.5 × target_kl），提前中止：

```python
approx_kl = (log_prob_old - log_prob_new).mean()
if approx_kl > 1.5 * target_kl:
    break
```

### 6. Value loss clip（可选）

和 actor clip 类似，限制 critic 每次更新幅度：

```python
v_clipped = v_old + torch.clamp(v_new - v_old, -eps, eps)
v_loss = torch.max((v_new - returns).pow(2), (v_clipped - returns).pow(2)).mean()
```

---

## 超参参考（CartPole-v1）

```python
hypers = {
    "GAMMA": 0.99,
    "GAE_lambda": 0.95,
    "eps_clip": 0.2,
    "K_epochs": 4,
    "rollout_steps": 2048,
    "minibatch_size": 64,
    "actor_LR": 3e-4,
    "critic_LR": 1e-3,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
}
```
