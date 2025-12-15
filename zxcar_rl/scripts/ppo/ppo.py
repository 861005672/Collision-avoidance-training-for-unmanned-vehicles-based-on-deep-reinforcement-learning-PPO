#! /home/zx567/Learning/pytorch_env/bin/python3


from torch.distributions import Categorical
import numpy as np
import torch
import copy
import math

import torch.nn.functional as F
import torch.nn as nn

import os
import copy
import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('zxcar_rl')

model_path = package_path+"/scripts/ppo/model/"																																										
runs_path = package_path+"/scripts/ppo/runs/"	


# 根据指定的网络形状构建网络
def build_net(layer_shape, activation, output_activation):
    layers = []
    for j in range(len(layer_shape)-1):
        act = activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]  
    return nn.Sequential(*layers)


# 策略网络 / Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hide_shape):
        super(Actor, self).__init__()
        layers = [state_dim] + list(hide_shape) + [action_dim]
        self.policy = build_net(layers, nn.Tanh, nn.Identity)  # nn.Identity 表示对输出层不做任何处理

    def forward(self, s):
        score = self.policy(s)
        soft_prob = F.softmax(score,dim=-1)
        return soft_prob


# 评价网络 / V值网络 / Critic / 状态值函数 v(s)
class Critic(nn.Module):
    def __init__(self, state_dim, hide_shape):
        super(Critic, self).__init__()
        layers = [state_dim] + list(hide_shape) + [1]
        self.value = build_net(layers, nn.Tanh, nn.Identity)

    def forward(self,s):
        v = self.value(s)
        return v


class PPO_agent():
    def __init__(self, timenow, model_name, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        # 创建保存模型的文件夹
        self.model_file_path = model_path + model_name + '--' + timenow
        if not os.path.exists(self.model_file_path) and self.write == True:
            os.mkdir(self.model_file_path)
            with open(self.model_file_path + '/说明.txt', 'w') as f:
                f.write('# 说明: \n')

        # 构建actor和critic网络
        self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,)*self.net_depth).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic = Critic(self.state_dim, (self.net_width,)*self.net_depth).to(self.dvc)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        

        # 创建经验回放区
        self.s_hoder = torch.zeros((self.train_exp_num, self.state_dim), dtype=torch.float32, device=self.dvc)
        self.a_hoder = torch.zeros((self.train_exp_num, 1), dtype=torch.int64, device=self.dvc)
        self.r_hoder = torch.zeros((self.train_exp_num, 1), dtype=torch.float32, device=self.dvc)
        self.s_next_hoder = torch.zeros((self.train_exp_num, self.state_dim), dtype=torch.float32, device=self.dvc)
        self.prob_a_hoder = torch.zeros((self.train_exp_num, 1), dtype=torch.float32, device=self.dvc)
        self.done_hoder = torch.zeros((self.train_exp_num, 1), dtype=torch.bool, device=self.dvc)
        self.dw_hoder = torch.zeros((self.train_exp_num, 1), dtype=torch.bool, device=self.dvc)

        # 熵正则项系数
        self.entropy_coef = self.entropy_init

    # 选择动作
    def select_action(self, state, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.dvc)
        with torch.no_grad():  # 这一行的作用: 在其代码块内，所有的张量计算都不会被 autograd 记录，也不会计算和存储梯度。
            probs = self.actor(state)
            if not deterministic:  # 如果不是确定性策略，则由策略网络输出动作 用于训练阶段保持agent的探索性
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
            else:
                action = torch.argmax(probs, dim=-1).squeeze()

            pi_a = probs.gather(0, action)


            # log日志记录probs
            log_path = os.path.expanduser('~/ppo_debug.log')
            with open(log_path, 'a') as f:
                f.write(str(probs.cpu().numpy()) + '\n')
                f.write(str(action) + '\n')
                f.write("=======================\n")

        return action.item(), pi_a



    def train(self):
        if self.entropy_coef <= self.entropy_min:self.entropy_coef = self.entropy_min

        # 熵系数衰减(探索率衰减)
        self.entropy_coef *= self.entropy_decay
        print(f"开始训练agent...")
        # 从hoder中取出值
        s = self.s_hoder
        a = self.a_hoder
        r = self.r_hoder
        s_next = self.s_next_hoder
        old_prob_a = self.prob_a_hoder
        done = self.done_hoder
        dw = self.dw_hoder

        # 计算GAE 优势函数
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            # dw 了的经验不加vs_
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            # 计算GAE
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps

        # 更新网络
        # 确保每次训练周期内，所有采集到的数据都能被用于 PPO 的 mini-batch 更新
        # 例如一个 long trajectory 有128个经验，batch_size = 64 那么 optim_iter_num = 2
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

        for _ in range(self.K_epochs):
            #Shuffle the trajectory, Good for training 打乱轨迹中样本的顺序，降低样本间相关性
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.dvc)
            # 取出打乱顺序后，长trajectory中的样本
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            # mini-batch ppo更新，每次更新optim_iter_num个经验
            # 在这里，optim_iter_num 保证了在每一个epochs内，经验回放区的每一个数据都会被拿来训练一次
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))
                # 更新actor网络
                prob = self.actor(s[index])
                entropy = Categorical(prob).entropy().mean()   # 熵项
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                # 更新critic网络
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:   # 只对权重参数进行正则化，不对偏置项进行正则化
                        # 计算所有权重参数的平方和，并乘以正则化系数 self.l2_reg，加到 Critic 的损失上。
                        # 这样做可以抑制模型参数过大，提高泛化能力
                        c_loss += param.pow(2).sum() * self.l2_reg  

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()
                
	# 储存一条经验
    def put_data(self, s, a, r, s_next, prob_a, done, dw, exp_idx):
        
        if not isinstance(s, torch.Tensor):
            s = torch.from_numpy(s).float().to(self.dvc)
        if not isinstance(s_next, torch.Tensor):
            s_next = torch.from_numpy(s_next).float().to(self.dvc)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.long, device=self.dvc)
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32, device=self.dvc)
        if not isinstance(prob_a, torch.Tensor):
            prob_a = torch.tensor(prob_a, dtype=torch.float32, device=self.dvc)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.bool, device=self.dvc)
        if not isinstance(dw, torch.Tensor):
            dw = torch.tensor(dw, dtype=torch.bool, device=self.dvc)   
            
        self.s_hoder[exp_idx,:] = s
        self.a_hoder[exp_idx,:] = a
        self.r_hoder[exp_idx,:] = r
        self.s_next_hoder[exp_idx,:] = s_next
        self.prob_a_hoder[exp_idx,:] = prob_a
        self.done_hoder[exp_idx,:] = done
        self.dw_hoder[exp_idx,:] = dw




    # 模型保存
    def save(self,algo,eps):
        # 保存Actor网络
        torch.save(self.actor.state_dict(), self.model_file_path + "/{}_{}_actor.pth".format(algo, eps))
        # 保存Critic网络
        torch.save(self.critic.state_dict(), self.model_file_path + "/{}_{}_critic.pth".format(algo, eps))


    # 模型载入
    def load(self, ModelName, ModelTime, algo, eps):
        actor_path = model_path + ModelName + '--' + ModelTime + "/{}_{}_actor.pth".format(algo, eps)
        critic_path = model_path + ModelName + '--' + ModelTime + "/{}_{}_critic.pth".format(algo, eps)
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.dvc, weights_only=True))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.dvc, weights_only=True))
