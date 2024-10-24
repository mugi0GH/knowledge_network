import torch
from Update_modules.memory.ExperienceReplay import ReplayMemory
from Update_modules.memory.AC_steps import batch
import copy
import numpy as np

class integrated_model():
    def __init__(self,actor,critic=None,hypers=None) -> None:
        self.actor = actor.to(hypers["DEVICE"])
        self.actor_oldPolicy = copy.deepcopy(self.actor)
        self.critic = critic.to(hypers["DEVICE"])
        self.hypers = hypers
        # self.batch = ReplayMemory(hypers["BATCH"])

        self.batch = batch(hypers["BATCH"],hypers['state_shape'])
        # self.lambda_kl = 1.0  # 初始的拉格朗日乘数

    def forward(self,state): # 返回 行为概率分布
        # state = torch.tensor(state).to(self.hyper["DEVICE"])
        '''actor'''
        state = torch.tensor(state).float().unsqueeze(0).to(self.hypers["DEVICE"])  # 将状态转化为张量
        probs = self.actor(state)  # 获取行为概率分布
        return probs
    
    def act(self,state): # 基于 行为概率分布 返回 行为
        probs = self.forward(state)

        action_dist = torch.distributions.Categorical(probs)  # 创建分布
        action = action_dist.sample()  # 采样行为

        # 记录每个行为的log概率到batch中
        # log_prob = action_dist.log_prob(action)
        # self.batch.log_probs[self.batch.ele_num] = log_prob

        return action.item()  # 返回行为
    
    def get_actorGrad(self):
        # 重新计算当前策略下的 log_probs
        probs = self.actor(self.batch.states)
        action_dist = torch.distributions.Categorical(probs)
        log_probs = action_dist.log_prob(self.batch.actions)
        
        # actor_loss = log(actor策略分布) * 优势函数值
        # 计算损失
        advantages = (self.batch.td_errors - self.batch.td_errors.mean()) / (self.batch.td_errors.std() + 1e-8)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # actor_loss = -(log_probs * self.batch.td_errors.detach()).mean()

        # 将熵添加到actor损失中：添加熵项可以鼓励策略维持探索：
        entropy = action_dist.entropy().mean()
        actor_loss -= 0.01 * entropy  # 熵系数可以调整

        # 对优势函数进行归一化（可选）
        # advantages = self.batch.td_errors
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算损失：log_prob * 优势函数
        # actor_loss = -(log_probs * advantages.detach()).mean()
        print(f"actor_loss: {actor_loss.item()}")
        
        # 清空梯度
        self.actor.optimizer.zero_grad()  # 如果未定义 optimizer，会报错
        # for param in self.actor.parameters():
        #     if param.grad is not None:
        #         param.grad.detach_()
        #         param.grad.zero_()
        
        # 计算新的梯度
        actor_loss.backward(retain_graph=False)
        
        # 提取梯度并展平
        flat_grads = torch.cat([p.grad.view(-1) for p in self.actor.parameters()])
        return flat_grads

    def compute_kl_divergence(self):
        with torch.no_grad():
            old_probs = self.actor_oldPolicy(self.batch.states)  # 旧策略的概率分布
        
        # 不使用 torch.no_grad()，以确保新策略计算时能够进行反向传播
        new_probs = self.actor(self.batch.states)  # 新策略的概率分布
        
        kl_divergence = torch.distributions.kl_divergence(
            torch.distributions.Categorical(probs=old_probs),
            torch.distributions.Categorical(probs=new_probs)
        )
        return kl_divergence.mean()  # 返回 KL 散度的均值

    
    def conjugate_gradient(self, g, max_iter=10, tol=1e-10):
        """
        共轭梯度法来求解 F^{-1}g，逼近自然梯度方向 v。

        参数:
        - fvp_fn: 计算费舍尔矩阵-向量积的函数。
        - g: 策略梯度向量。
        - max_iter: 最大迭代次数。
        - tol: 残差容忍度。
        
        返回:
        - v: 近似求解的 F^{-1} g，即自然梯度方向。
        """

        # 初始解，x 是要逼近的自然梯度方向 v, 初始时为零向量
        x = torch.zeros_like(g)
        
        # 初始残差 r = g - F*x, 由于初始时 x=0，所以 r = g
        r = g.clone()
        
        # 初始搜索方向 p，等于初始残差 r
        p = r.clone()

        # r 和 r 的内积，用于共轭梯度更新时计算步长
        r_dot_old = torch.dot(r, r)
        
        # 迭代次数上限
        for _ in range(max_iter):
        # i = 0
        # r_dot_new = tol+1
        # while r_dot_new>tol:
            # Step 1: 计算费舍尔矩阵-向量积 Fp, fvp_fn 是 Fisher 矩阵与向量的乘积函数
            fvp = self.fisher_vector_product(p)
            
            fvp_dot_p = torch.dot(p, fvp)
            if fvp_dot_p == 0:
                fvp_dot_p = 1e-8  # 避免除零

            # Step 2: 计算步长系数 alpha，确保步长在当前搜索方向上是合理的
            alpha = r_dot_old / fvp_dot_p
            
            # 防止步长计算不稳定，限制 fvp_dot_p 不为负或过小
            if alpha <= 0:
                alpha = max(alpha, 1e-10)

            # Step 3: 更新解 x (即当前的自然梯度方向的近似)
            x += alpha * p
            # x = x + (alpha * p)
            
            # Step 4: 更新残差 r (表示剩余误差)
            # r -= alpha * fvp
            r = r - (alpha * fvp).clone()
            
            # 计算新的残差内积，用于判断收敛性
            r_dot_new = torch.dot(r, r)
            
            # 如果残差足够小，说明已经逼近解，可以提前停止迭代
            # print('tol',r_dot_new,'\t',tol)
            if r_dot_new < tol:
                break

            # Step 5: 计算共轭梯度系数 beta
            beta = r_dot_new / r_dot_old

            # Step 6: 更新搜索方向 p, 新搜索方向是基于残差 r 和旧搜索方向 p 的线性组合
            p = r + beta * p.clone()

            # 更新 r_dot_old 为下一次迭代使用
            r_dot_old = r_dot_new

            # i+=1
        # 返回最终的解 x，它是 F^{-1}g 的近似解，即自然梯度方向 v
        return x

    def fisher_vector_product(self, v):
        """
        计算 Fisher 矩阵与向量 v 的乘积 Fv。
        
        参数:
        - kl: KL 散度
        - params: 策略的参数
        - v: 向量 v
        
        返回:
        - fvp: Fisher 矩阵与 v 的乘积 Fv
        """

        # 隐式计算：通过 KL 散度的二阶导数, 来近似 Fisher 矩阵与向量的乘积 "Fv"，而不显式构造 Fisher 矩阵。这个方法适用于模型规模较大的场景，尤其是在神经网络参数较多的情况下，显式构建和存储 Fisher 矩阵会非常昂贵。

        # Step 1: 计算 KL 散度对参数的梯度
        kl = self.compute_kl_divergence()

        params = list(self.actor.parameters())  # 获取"新"策略的参数
        # kl_grad = torch.autograd.grad(kl, params, create_graph=True, retain_graph= True)
        # 在 fisher_vector_product 中计算 KL 散度的梯度时，可能会遇到某些参数的梯度为 None。这是因为在计算梯度的过程中，可能有些参数未参与计算图。你可以通过设置 allow_unused=True 来解决这个问题，避免错误发生。
        kl_grad = torch.autograd.grad(kl, params, create_graph=True, retain_graph=True, allow_unused=True)
        
        # Step 2: 将梯度与向量 v 做内积, 在提取梯度时，跳过为 None 的梯度。
        kl_grad_vector = torch.cat([g.view(-1) for g in kl_grad if g is not None])  # 将梯度展平为向量
        kl_grad_v = torch.dot(kl_grad_vector, v)  # 计算内积
        
        # Step 3: 对内积结果再次求梯度，得到 Fv (Hessian-Vector Product)
        fvp = torch.autograd.grad(kl_grad_v, params, retain_graph= True,allow_unused=True)
        
        # 将结果拼接为一个向量
        fvp = torch.cat([g.contiguous().view(-1) for g in fvp])

        # 加入阻尼因子，防止数值不稳定
        damping = 1e-5
        fvp = fvp + damping * v.clone()
        # 显式计算
        """
        damping = 0.1
        # 将所有参数的梯度展平为一个大张量
        flat_grads = torch.cat([g.view(-1) for g in grads])  # 将所有梯度展平并拼接

        # 计算外积，得到费舍尔信息矩阵(Fisher matrix)的一个样本(初始Fisher矩阵) + 阻尼项，避免数值不稳定性; torch.eye单位矩阵;
        fisher_matrix = torch.outer(flat_grads, flat_grads)
        fisher_matrix += damping * torch.eye(flat_grads.size(0)).to(self.hypers["DEVICE"])
        fisher_matrix /= len(self.batch.states)
        """
        return fvp
    
    def get_flat_params_from_old(self):
        return torch.cat([param.view(-1) for param in self.actor_oldPolicy.parameters()])

    def set_flat_params_to_new(self,flat_params):
        idx = 0
        for param in self.actor.parameters():
            param_length = param.numel()
            param.data.copy_(flat_params[idx:idx + param_length].view(param.size()))
            idx += param_length

    # def line_search_with_kl(self, v, alpha=1.0, beta=0.5, max_iter=10):
    #     max_kl = self.hypers['delta']
    #     old_params = self.get_flat_params_from_old()  # 获取旧策略的展平参数

    #     for i in range(max_iter):
    #         new_params = old_params + alpha * v  # 根据当前步长和方向更新策略参数
    #         self.set_flat_params_to_new(new_params)  # 将新参数赋值给 新策略

    #         # 计算新策略和旧策略的 KL 散度
    #         kl = self.compute_kl_divergence()

    #         # 更新拉格朗日乘数
    #         self.update_lambda(kl)

    #         # 计算目标函数，包含 KL 散度惩罚项
    #         objective = self.compute_objective(new_params) - self.lambda_kl * (kl - max_kl)

    #         # 检查 KL 散度是否满足约束
    #         if kl <= max_kl:
    #             return True

    #         # 如果 KL 散度超出约束，缩小步长
    #         alpha *= beta

    #         if alpha < 1e-8:  # 设置最小步长限制
    #             break

    #     # 如果线性搜索失败，返回原始参数
    #     self.set_flat_params_to_new(old_params)
    #     return False

    def line_search_with_kl(self,v, alpha=1.0, beta=0.5, max_iter=10):
        """
        带有 KL 散度约束的线性搜索，确保更新后的策略不会偏离旧策略太多。
        
        参数:
        - policy: 当前策略的网络
        - old_policy: 旧策略，用于计算 KL 散度
        - loss_fn: 损失函数，用于评估策略性能
        - kl_fn: KL 散度计算函数
        - v: 自然梯度方向
        - max_kl: KL 散度的最大允许值
        - alpha: 初始步长
        - beta: 步长缩小因子
        - max_iter: 最大线性搜索迭代次数
        
        返回:
        - new_params: 更新后的策略参数
        """
        max_kl= self.hypers['delta']
        old_params = self.get_flat_params_from_old()  # 获取旧策略的展平参数

        for i in range(max_iter):
            new_params = old_params + alpha * v  # 根据当前步长和方向更新策略参数
            self.set_flat_params_to_new(new_params)# 将新参数赋值给 新策略
            # 计算新策略和旧策略的 KL 散度
            kl = self.compute_kl_divergence()

            # 检查 KL 散度是否满足约束
            # print('kl',kl,'\t',max_kl)
            if kl <= max_kl:  # 如果 KL 散度满足约束，退出线性搜索
                return True
            
            # 如果 KL 散度超出约束，缩小步长
            alpha *= beta
            
            if alpha < 1e-8:  # 设置最小步长限制
                break

        
        # 如果线性搜索失败，返回原始参数
        self.set_flat_params_to_new(old_params)
        return False

    # def update_lambda(self, kl_divergence):
    #     delta = self.hypers['delta']  # 允许的最大 KL 散度
    #     alpha_lambda = 0.01  # 拉格朗日乘数的学习率

    #     # 更新拉格朗日乘数
    #     self.lambda_kl += alpha_lambda * (kl_divergence - delta)

    #     # 确保拉格朗日乘数不为负
    #     self.lambda_kl = max(self.lambda_kl, 0.0)


    def optimize(self,state, action, next_state, reward,done):
        # batch
        if self.batch.__len__()< self.hypers['BATCH']-1:
            
            # 1. Critic 计算 TD 误差 （优势函数）
            # td_error = self.critic.optimize(state, next_state, reward, done) # TD 误差 Gt - Vt

            # self.batch.push(state, action, next_state, reward,done,td_error) # 存入batch中
            self.batch.push(state, action, next_state, reward,done) # 存入batch中
            return

        self.batch.push(state, action, next_state, reward,done) # 存入batch中

        # 优化 Critic
        self.batch.td_errors = self.critic.optimize(self.batch.states, self.batch.next_states, self.batch.rewards, self.batch.dones) # G_t

        # 使用运行平均值和标准差来归一化奖励。这将有助于actor-critic网络稳定学习。
        self.batch.rewards = (self.batch.rewards - self.batch.rewards.mean()) / (self.batch.rewards.std() + 1e-8)


        if self.hypers['TRPO']:
            # actor有 新权重, 旧策略不变。
            grads = self.get_actorGrad() # g

            # 计算方向 v
            v = self.conjugate_gradient(g=grads,max_iter=self.hypers['max_iter'])

            # KL 散度, 若散度为0，说明权重完全一致。
            # 线性搜索 查找 步长
            success = self.line_search_with_kl(v,beta=self.hypers['beta'],max_iter=self.hypers['max_iter'])
            if success:
                self.actor_oldPolicy = copy.deepcopy(self.actor)
                print("Line search success.")
            else:
                print("Line search failed, reverting to old parameters.")
        elif self.hypers['VPG']:
            self.get_actorGrad()
            self.actor.optimizer.step()
        
        # 拉格朗日法 查找 步长
        self.batch.reset()