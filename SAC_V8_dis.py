import multiprocessing
import os
import random
import time
from collections import deque

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal

from env import Env  # 导入自定义环境
from utils import weights_init_

torch.autograd.set_detect_anomaly(True)


import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

# Replay Buffer 
class ReplayBuffer:
    def __init__(self, max_size, env):
        self.buffer = deque(maxlen=max_size)
        self.env = env
        self.now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    
    def add(self, state, action, reward, next_state, done, info):
        self.buffer.append((state, action, reward, next_state, done, info))
    
    def sample(self, batch_size):
        # get the batch data from new data
        success_ratio = 0.3
        latest_ratio = 0.2
        random_ratio = 0.5
        num_success = round(success_ratio * batch_size)
        num_latest = round(latest_ratio * batch_size)
        num_random = round(random_ratio * batch_size)

        # initial the sampled data
        sampled_success_experiences = []
        success_indices = [idx for idx, experience in enumerate(self.buffer) 
                   if experience[-1].get("reaction") == "success"]
        actual_success = min(num_success, len(success_indices))
        if actual_success > 0:
            sampled_success = random.sample(success_indices, actual_success)
            sampled_success_experiences = [self.buffer[idx] for idx in sampled_success]
        else:
            sampled_success_experiences = []
            actual_success = 0
        sampled_latest_experiences = []
        if num_latest > 0:
            sampled_latest_experiences = list(self.buffer)[-num_latest:]
        remaining = batch_size - actual_success - len(sampled_latest_experiences)
        if remaining > 0:
            exclude_indices = set()
            try:
                exclude_indices.update(sampled_success)
            except:
                pass
            exclude_indices.update(range(len(self.buffer) - num_latest, len(self.buffer)))
            
            available_indices = list(set(range(len(self.buffer))) - exclude_indices)
            
            if len(available_indices) < remaining:
                sampled_random_experiences = random.choices(self.buffer, k=remaining)
            else:
                sampled_random = random.sample(available_indices, remaining)
                sampled_random_experiences = [self.buffer[idx] for idx in sampled_random]
        else:
            sampled_random_experiences = []
        
        batch = sampled_success_experiences + sampled_latest_experiences + sampled_random_experiences

        if len(batch) < batch_size:
            additional = batch_size - len(batch)
            additional_samples = random.choices(self.buffer, k=additional)
            batch += additional_samples


        batch = random.sample(self.buffer, batch_size)
        random.shuffle(batch)
        states, actions_index, rewards, next_states, dones, info = zip(*batch)
        
        return (np.array(states), np.array(actions_index), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    
    def size(self):
        return len(self.buffer)

    def plot_train(self, hyperparameters):
        # plot the action and plot the reward
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(2, 2, height_ratios=[5, 3], width_ratios=[5, 5])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        action_bound = [self.env.action_space.low, self.env.action_space.high]
        low, high = action_bound

        scale = (high - low) / 2
        bias = (high + low) / 2

        ax1.set_title("x, y Distribution")
        ax1.set_xlim(-50,50)
        ax1.set_ylim(-50,50)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_title("v, a Distribution")
        ax2.set_xlim(action_bound[0][2],action_bound[1][2])
        ax2.set_ylim(action_bound[0][3],action_bound[1][3])
        ax2.set_xlabel("v")
        ax2.set_ylabel("a")
        data_x, data_y, colors_xy = [], [], []
        data_v, data_a, colors_va = [], [], []
        reward_list = []
        for state, action, reward, next_state, done, info in self.buffer:
            if self.env.polar_space:
                l, theta, v, a = action
                x = l * np.cos(np.radians(theta))
                y = l * np.sin(np.radians(theta))
            else:
                x, y, v, a = action
            x = x * scale[0] + bias[0]
            y = y * scale[1] + bias[1]
            v = v * scale[2] + bias[2]
            a = a * scale[3] + bias[3]

            if info["reaction"] == "success":
                color = (0, 1, 0, 0.5)  # 半透明绿色
            elif info["reaction"] == "failure":
                color = (1, 0, 0, 0.5)  # 半透明红色
            else:  # 无变化
                color = (0.5, 0.5, 0.5, 0.5)  # 半透明灰色
            data_x.append(x)
            data_y.append(y)
            colors_xy.append(color)
            data_v.append(v)
            data_a.append(a)
            colors_va.append(color)
            reward_list.append(reward)
        ax1.scatter(data_x, data_y, c=colors_xy, alpha=0.5)
        ax2.scatter(data_v, data_a, c=colors_va, alpha=0.5)
        ax3.plot(reward_list, color='blue')
        plt.tight_layout()

        if not os.path.exists("./train_result_plot"):
            os.makedirs("./train_result_plot")
        # add the hyperparameters to the figure name
        str_hyperparameters = '_'.join([f"{key}={value}" for key, value in hyperparameters.items()])
        plt.savefig(f"train_result_plot/dis_action_{self.now}_{str_hyperparameters}.png")
    
    # save the buffer data as a txt file
    def save(self):
        # now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        with open(f"train_result_plot/dis_buffer_data_{self.now}.txt", "w") as f:
            for state, action, reward, next_state, done, info in self.buffer:
                f.write(f"state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, done: {done}, info: {info}\n")
        print("Save the buffer data successfully!")

# Initialize weights function
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

# Discrete Policy Network
class DiscretePolicyNet(nn.Module):
    def __init__(self, input_dim, action_dims, hidden_dim=256):
        super(DiscretePolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for action_dim in action_dims
        ])
        self.apply(weights_init_)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = [head(x) for head in self.action_heads]
        return action_logits
    
    def sample_action(self, x):
        action_logits = self.forward(x)
        action_probs = [F.softmax(logit, dim=-1) for logit in action_logits]
        actions_index = []
        log_probs = []
        for prob in action_probs:
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()
            actions_index.append(action)
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
        actions_index = torch.stack(actions_index, dim=-1)  # [batch_size, num_action_dims]
        log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1, keepdim=True)  # [batch_size, 1]
        return actions_index, log_probs

# Discrete Critic Network
class DiscreteCriticNet(nn.Module):
    def __init__(self, input_dim, action_dims, hidden_dim=256):
        super(DiscreteCriticNet, self).__init__()
        self.input_dim = input_dim
        self.action_dims = action_dims
        self.fc1 = nn.Linear(input_dim + sum(action_dims), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)
    
    def forward(self, state, actions_index):
        one_hot_actions = []
        for i, action_dim in enumerate(self.action_dims):
            one_hot = F.one_hot(actions_index[:, i], num_classes=action_dim).float()
            one_hot_actions.append(one_hot)
        one_hot_actions = torch.cat(one_hot_actions, dim=-1)*50  # [batch_size, sum(action_dims)]
        x = torch.cat([state, one_hot_actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q_value(x)
        return q

# SAC Agent for Discrete Actions
class SACAgent:
    def __init__(self, env: Env, buffer_size=10000, 
                 batch_size=64, gamma=0.99, tau=0.005, alpha=0.2, lr=0.0003,
                 action_dims=[20, 20, 20, 20]):
        self.hyperparameters = {
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "tau": tau,
            "alpha": alpha,
            "lr": lr
        }
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dims = action_dims
        self.num_action_dims = len(action_dims)
        self.action_dim_total = sum(action_dims)

        self.action_bound = [env.action_space.low, env.action_space.high]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.target_entropy = -sum(action_dims)
    
        self.replay_buffer = ReplayBuffer(buffer_size, env)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
    
        self.epsilon = 1e-6

        low, high = self.action_bound

        self.action_scale = torch.tensor((high - low) / 2, dtype=torch.float32, device=self.device)
        self.action_bias = torch.tensor((high + low) / 2, dtype=torch.float32, device=self.device)
    
        # Initialize networks
        self.policy_net = DiscretePolicyNet(self.state_dim, self.action_dims).to(self.device)
        self.q_net1 = DiscreteCriticNet(self.state_dim, self.action_dims).to(self.device)
        self.q_net2 = DiscreteCriticNet(self.state_dim, self.action_dims).to(self.device)
        self.target_q_net1 = DiscreteCriticNet(self.state_dim, self.action_dims).to(self.device)
        self.target_q_net2 = DiscreteCriticNet(self.state_dim, self.action_dims).to(self.device)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
    
        # Initialize entropy temperature parameter
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
    
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr, weight_decay=1e-4)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr, weight_decay=1e-4)
    
    def one_hot_encode(self, state):
        """将状态转换为独热编码，具体实现根据环境而定"""
        # 示例实现，需根据具体环境调整
        # 假设状态是一个整数，代表分子状态
        molecule_state = int(state) + 1  # 假设状态在 0,1,2
        one_hot = np.zeros(1)
        one_hot[0] = molecule_state  # 独热编码
        return np.array(one_hot)
    
    # def remap_action(self, actions_index):
    #     """将离散动作索引映射到实际动作值"""
    #     mapped_actions = []
    #     for i in range(self.num_action_dims):
    #         # 假设每个动作维度的值在 [-1, 1] 之间均匀分布
    #         discrete_levels = np.linspace(-1, 1, self.action_dims[i])
    #         # mapped = discrete_levels[actions_index[:, i].cpu().numpy()]
    #         mapped = discrete_levels[actions_index[i].cpu().numpy()]
    #         mapped = self.action_bias + self.action_scale * mapped
    #         mapped_actions.append(mapped)
    #     mapped_actions = np.stack(mapped_actions, axis=-1)  # [batch_size, num_action_dims]
    #     return mapped_actions
    
    def remap_action(self, actions_index):
        """将离散动作索引映射到实际动作值"""
        mapped_actions = []
        for i in range(self.num_action_dims):
            # 假设每个动作维度的值在 [-1, 1] 之间均匀分布
            discrete_levels = np.linspace(-1, 1, self.action_dims[i])
            # mapped = discrete_levels[actions_index[:, i].cpu().numpy()]
            mapped = discrete_levels[actions_index[i].cpu().numpy()]
            # mapped = self.action_bias + self.action_scale * mapped
            mapped_actions.append(mapped)
        mapped_actions = torch.tensor(mapped_actions, device=self.device)
        mapped_actions = self.action_bias + self.action_scale * mapped_actions
        # mapped_actions = np.stack(mapped_actions, axis=-1)  # [batch_size, num_action_dims]
        return mapped_actions
    
    # def reverse_map(self, mapped_actions):
    #     """将实际动作值映射到离散动作索引"""
    #     actions_index = []
    #     action  = torch.round((mapped_actions - self.action_bias) / self.action_scale)


            

    def select_action(self, states, deterministic=False, detach_action=True):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float().to(self.device)
        elif isinstance(states, torch.Tensor):
            states = states.to(self.device)
        
        if deterministic:
            # 选择每个动作维度概率最大的动作
            action_logits = self.policy_net(states)
            actions_index = []
            for logits in action_logits:
                action = torch.argmax(logits, dim=-1)
                actions_index.append(action)
            actions_index = torch.stack(actions_index, dim=-1)  # [batch_size, num_action_dims]
            log_prob = torch.zeros(actions_index.size(0), 1).to(self.device)
        else:
            # 随机采样动作
            actions_index, log_prob = self.policy_net.sample_action(states)
        
        actions = self.remap_action(torch.tensor(actions_index).to(self.device))
        
        if detach_action:
            return actions.cpu().numpy(), log_prob.detach(), actions_index.cpu().numpy()
        return actions, log_prob, actions_index
    
    def train(self, num_steps, render=False, visualize=False):
        state = self.env.reset()
        state = self.one_hot_encode(state)
        
        # plot the action
        if visualize:
            queue = multiprocessing.Queue(50)
            plot_process = multiprocessing.Process(target=self.plot_actions, args=(queue,self.env))
            plot_process.start()
        
        for step in range(num_steps):
            # 选择动作
            actions, _, actions_index = self.select_action(state, detach_action=True)
            # remapped_action = self.remap_action(torch.tensor(action_indices).to(self.device))
            
            # 与环境交互
            next_state, reward, done, info = self.env.step(actions)
            # add the actions_index into the info
            info["actions_index"] = actions_index
            print(f"Reward: {reward}")
            next_state = self.one_hot_encode(next_state)
            self.replay_buffer.add(state, actions_index, reward, next_state, done, info)
    
            if render:
                self.env.render()
                self.env.clock.tick(60)
            
            if visualize:
                
                if queue.full():  # 如果队列满了，则等待一段时间
                    queue.get()
                queue.put((state.copy(), actions.copy(), reward, next_state.copy(), done, info.copy()))

            if self.replay_buffer.size() > self.batch_size:
                self.update()
    
            if step % 1000 == 0:
                self.save_model(step)
    
            if done:
                state = self.one_hot_encode(self.env.reset())
            else:
                state = next_state
            
        if visualize:
            queue.put(None)
            plot_process.join()

        self.replay_buffer.plot_train(self.hyperparameters)
        self.replay_buffer.save()
            
        # 训练结束后的步骤
    
    def update(self):
        # get experience from replay buffer
        states, actions_index, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
    
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_index = torch.tensor(actions_index, dtype=torch.long, device=self.device)  # 离散动作索引

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
    
        # Q-loss
        with torch.no_grad():
            next_actions, next_state_log_pi = self.policy_net.sample_action(next_states)
            target_q1 = self.target_q_net1(next_states, next_actions)
            target_q2 = self.target_q_net2(next_states, next_actions)
            q_min = torch.min(target_q1, target_q2) - self.alpha * next_state_log_pi
            q_target = rewards + (1 - dones) * self.gamma * q_min
    
        current_q1 = self.q_net1(states, actions_index)
        current_q2 = self.q_net2(states, actions_index)
    
        q_loss1 = F.mse_loss(current_q1, q_target)
        q_loss2 = F.mse_loss(current_q2, q_target)
        q_loss = q_loss1 + q_loss2
    
        self.q_optimizer1.zero_grad()
        self.q_optimizer2.zero_grad()
        q_loss.backward()

    
        # Policy-loss
        new_actions, log_prob = self.policy_net.sample_action(states)
        q1_new = self.q_net1(states, new_actions)
        q2_new = self.q_net2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - min_q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
    
        # α-loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()


        # Update all networks
        self.q_optimizer1.step()
        self.q_optimizer2.step()
        self.policy_optimizer.step()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().detach()
    
        # Soft update target networks
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self, step):
        # get the current date and time
        now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        if not os.path.exists(f"models/{now}"):
            os.makedirs(f"models/{now}")
        torch.save(self.policy_net.state_dict(), f"models/{now}/policy_net_{step}.pth")
        torch.save(self.q_net1.state_dict(), f"models/{now}/q_net1_{step}.pth")
        torch.save(self.q_net2.state_dict(), f"models/{now}/q_net2_{step}.pth")

    @classmethod
    def plot_actions(self,queue,env):
        """绘图进程：消费队列中的数据并更新图表"""
        # 初始化绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title("x, y Distribution")
        action_bound = [env.action_space.low, env.action_space.high]
        # set the range of x and y, according to the action_bound
        ax1.set_xlim(action_bound[0][0],action_bound[1][0])
        ax1.set_ylim(action_bound[0][1],action_bound[1][1])
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_title("v, a Distribution")
        # set the range of v and a, according to the action_bound
        ax2.set_xlim(action_bound[0][2],action_bound[1][2])
        ax2.set_ylim(action_bound[0][3],action_bound[1][3])
        ax2.set_xlabel("v")
        ax2.set_ylabel("a")

        data_x, data_y, colors_xy = [], [], []
        data_v, data_a, colors_va = [], [], []
        # pop data_x data_y colors_xy data_v data_a colors_va if the length is more than 50

        while True:
            item = queue.get()
            if item is None:  # 终止信号
                break

            state, action, reward, next_state, done, info = item
            if env.polar_space:
                l, theta, v, a = action # theta in 0-360
                x = l * np.cos(np.radians(theta))
                y = l * np.sin(np.radians(theta))
            else:
                x, y, v, a = action

            # 根据反应结果设置颜色
            if info["reaction"] == "success":
                color = (0, 1, 0, 0.5)  # 半透明绿色
            elif info["reaction"] == "failure":
                color = (1, 0, 0, 0.5)  # 半透明红色
            else:  # 无变化
                color = (0.5, 0.5, 0.5, 0.5)  # 半透明灰色

            # 更新数据
            data_x.append(x)
            data_y.append(y)
            colors_xy.append(color)
            data_v.append(v)
            data_a.append(a)
            colors_va.append(color)
            # pop data_x data_y colors_xy data_v data_a colors_va if the length is more than 50
            if len(data_x) > 50:
                data_x.pop(0)
                data_y.pop(0)
                colors_xy.pop(0)
                data_v.pop(0)
                data_a.pop(0)
                colors_va.pop(0)
            # 更新图表
            # clear the ax1 and ax2
            ax1.clear()
            ax2.clear()
            ax1.set_xlim(-50,50)
            ax1.set_ylim(-50,50)
            ax2.set_xlim(action_bound[0][2],action_bound[1][2])
            ax2.set_ylim(action_bound[0][3],action_bound[1][3])
            ax1.scatter(data_x, data_y, c=colors_xy, alpha=0.5)
            ax2.scatter(data_v, data_a, c=colors_va, alpha=0.5)
            plt.pause(0.01)  # 实时刷新


if __name__ == "__main__":
        env = Env(polar_space=False)
        agent = SACAgent(env)
        agent.train(10000,visualize=True)



