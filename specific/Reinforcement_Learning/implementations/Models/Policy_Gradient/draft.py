import torch
import copy
from Update_modules.memory.ExperienceReplay import ReplayMemory
from Update_modules.memory.AC_steps import batch
import numpy as np

class integrated_model():
    def __init__(self, actor, critic=None, hypers=None):
        self.actor = actor.to(hypers["DEVICE"])
        self.actor_old = copy.deepcopy(actor).to(hypers["DEVICE"])
        self.critic = critic.to(hypers["DEVICE"])
        self.hypers = hypers
        self.batch = batch(hypers["BATCH"], hypers['state_shape'])
    
    def forward(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(self.hypers["DEVICE"])
        probs = self.actor(state)
        return probs
    
    def act(self, state):
        probs = self.forward(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        self.batch.log_probs[self.batch.ele_num] = log_prob
        return action.item()
    
    def get_actorGrad(self):
        actor_loss = (-self.batch.log_probs * self.batch.td_errors.detach()).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        grads = []
        for param in self.actor.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        flat_grads = torch.cat(grads)
        return flat_grads
    
    def conjugate_gradient(self, Fvp_fn, g, max_iter=10, tol=1e-10):
        x = torch.zeros_like(g)
        r = g.clone()
        p = g.clone()
        r_dot_old = torch.dot(r, r)
        for _ in range(max_iter):
            Avp = Fvp_fn(p)
            alpha = r_dot_old / torch.dot(p, Avp)
            x += alpha * p
            r -= alpha * Avp
            r_dot_new = torch.dot(r, r)
            if r_dot_new < tol:
                break
            beta = r_dot_new / r_dot_old
            p = r + beta * p
            r_dot_old = r_dot_new
        return x
    
    def natural_gradient_update(self, g, Fvp_fn, delta):
        v = self.conjugate_gradient(lambda p: Fvp_fn(p, g, self.batch.states), g)
        Fv = Fvp_fn(v, g, self.batch.states)
        step_size = torch.sqrt(2 * delta / torch.dot(v, Fv)) # 步长 alpha
        params_new = g + step_size * v

        index = 0
        for param in self.actor.parameters():
            param_length = param.numel()
            param.data.copy_(params_new[index:index + param_length].view(param.size()))
            index += param_length
    
    def fisher_vector_product(self, p, params, states):
        damping = 0.1
        kl = self.compute_kl_divergence(states)
        grads = torch.autograd.grad(kl, params, create_graph=True)
        flat_grads = torch.cat([grad.view(-1) for grad in grads])
        kl_p = torch.dot(flat_grads, p)
        grads_kl_p = torch.autograd.grad(kl_p, params, retain_graph=True)
        fisher_vec_prod = torch.cat([grad.contiguous().view(-1) for grad in grads_kl_p])
        return fisher_vec_prod + damping * p
    
    def compute_kl_divergence(self, states):
        with torch.no_grad():
            old_probs = self.actor_old(states)
        new_probs = self.actor(states)
        kl_divergence = torch.distributions.kl_divergence(
            torch.distributions.Categorical(probs=old_probs),
            torch.distributions.Categorical(probs=new_probs)
        )
        return kl_divergence.mean()
    
    def optimize(self, state, action, next_state, reward, done):
        if self.batch.__len__() < self.hypers['BATCH']-1:
            td_error = self.critic.optimize(state, next_state, reward, done)
            self.batch.push(state, action, next_state, reward, done, td_error)
            return
        
        td_error = self.critic.optimize(state, next_state, reward, done)
        self.batch.push(state, action, next_state, reward, done, td_error)

        flat_grads = self.get_actorGrad() # g

        self.natural_gradient_update(flat_grads, self.fisher_vector_product, delta=self.hypers['delta'])

        self.actor_old.load_state_dict(self.actor.state_dict())

        self.batch.reset()
