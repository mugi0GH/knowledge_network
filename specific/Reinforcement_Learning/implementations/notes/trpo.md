# TRPO（Trust Region Policy Optimization）

## 核心思想

普通 Policy Gradient 步长难以控制，可能导致策略崩溃。TRPO 引入 KL 散度约束，限制每次更新前后策略的变化幅度：

```
maximize  L(θ)        # 策略目标
subject to KL(π_old || π_new) ≤ δ
```

## 自然梯度方向

普通梯度 `g` 在参数空间走直线，但策略空间是弯曲的——不同参数维度对策略的影响不均匀。
Fisher 信息矩阵 `F` 描述这种曲率，自然梯度用 `F⁻¹` 矫正方向：

```
v = F⁻¹g
```

- `g`：普通策略梯度
- `F`：Fisher 信息矩阵（= KL 散度的 Hessian）
- `v`：自然梯度方向（在策略空间中"最陡"的方向）

`F` 矩阵太大无法直接求逆，实际用**共轭梯度（CG）**求解 `Fv = g`。

### Fisher-Vector Product（FVP）

CG 只需要 `Fv` 的乘积，不需要显式构造 `F`：

```python
# 用两次自动微分实现
kl_grad = autograd.grad(kl, params, create_graph=True)
fvp = autograd.grad(kl_grad · v, params, allow_unused=True)
# 注意：allow_unused=True 可能返回 None，需替换为 zeros_like(p)
```

## 初始步长

KL 散度的二阶泰勒近似：

```
KL(π_old || π_new) ≈ ½·(αv)ᵀ F (αv)
                    = ½α²·vᵀFv
                    = ½α²·vᵀg      # 因为 Fv = g
                    = ½α²·gᵀv
```

约束 `KL ≤ δ` 反解步长：

```
½α²·gᵀv ≤ δ
α = sqrt(2δ / gᵀv)
```

这是**解析解，不是启发式**——是在二阶近似下恰好满足 KL 约束的最大步长。

### gᵀv 的作用

```
gᵀv 大 → 当前方向曲率大，策略变化快 → α 小
gᵀv 小 → 当前方向曲率小，策略变化慢 → α 大
```

δ 是超参（固定），gᵀv 每次都不同，α 随之自动调整。

## Line Search

初始 α 基于二阶近似，可能不精确。Line Search 验证真实 KL：

```
for i in range(max_iter):
    θ_new = θ + α·v
    if KL(π_old || π_new) ≤ δ and L(θ_new) > L(θ):
        accept
    else:
        α *= β      # 典型 β=0.5，指数回退
```

## 完整流程

```
1. 采集 batch 数据（states, actions, rewards, ...）
2. 计算 advantage（用 critic，no_grad）
3. 算策略梯度 g
4. 用 CG 解 Fv = g，得自然梯度方向 v
5. 算初始步长 α = sqrt(2δ / gᵀv)
6. Line search：验证真实 KL ≤ δ，不满足则 α *= β 回退
7. 更新 θ = θ + α·v
8. 更新 critic
```

## VPG vs TRPO

| | VPG | TRPO |
|--|--|--|
| **更新方向** | 普通梯度 `g` | 自然梯度 `F⁻¹g` |
| **步长控制** | 固定学习率 | KL 约束 + line search |
| **稳定性** | 容易步长过大崩溃 | 有理论保证的单调改进 |
| **计算成本** | 低 | 高（CG + line search） |

## 超参参考（CartPole-v1）

```python
hypers = {
    "GAMMA": 0.99,
    "critic_LR": 1e-3,
    "BATCH": 1000,
    "v_loss_iter": 30,
    "GAE": True,
    "TD_lambda": 0.95,
    "delta": 0.01,       # KL 约束上限
    "beta": 0.5,         # line search 回退系数
    "max_iter": 10,      # line search 最大迭代次数
    "update_per_episode": False,
}
```

## 常见实现陷阱

1. **FVP 中的 None 梯度**：`allow_unused=True` 可能返回 None，需替换为 `zeros_like(p)`
2. **初始步长用 1.0**：没有数学依据，应用 `sqrt(2δ/gᵀv)`
3. **Advantage shape 不匹配**：actions 存为 `[BATCH, 1]` 导致广播出 outer product，应存为 `[BATCH]`
4. **Reward normalization 破坏信号**：CartPole reward 全为 1.0，归一化后全为 0。只在 advantage 上归一化
5. **Advantage/critic 耦合**：先更新 critic 再算 advantage，actor 拿到的是近零残差
