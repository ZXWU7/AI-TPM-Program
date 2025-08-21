import multiprocessing
import os
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal

from env import Env  # 导入自定义环境


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# Replay Buffer 
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    
    def add(self, state, action, reward, next_state, done, info):
        self.buffer.append((state, action, reward, next_state, done, info))
    
    def sample(self, batch_size):
        # get the batch data from new data
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, info = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    def sample_tip(self, batch_size):
        # get the batch data from new data
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, info = zip(*batch)
        actions = np.array(actions)[:, -2:]  # 只取最后两个维度
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    def size(self):
        return len(self.buffer)

    def plot_train(self):
        # plot the action and plot the reward
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.set_title("x, y Distribution")
        ax1.set_xlim(-100, 100)       # x range
        ax1.set_ylim(-100, 100)       # y range
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_title("v, a Distribution")
        ax2.set_xlim(3, 4)         # v range
        ax2.set_ylim(0.3, 1.7)         # a range
        ax2.set_xlabel("v")
        ax2.set_ylabel("a")
        data_x, data_y, colors_xy = [], [], []
        data_v, data_a, colors_va = [], [], []
        reward_list = []
        for state, action, reward, next_state, done, info in self.buffer:
            x, y, v, a = action
            x = 0
            y = 0
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

        if not os.path.exists("./train_result_plot"):
            os.makedirs("./train_result_plot")
        plt.savefig(f"train_result_plot/action_{self.now}.png")
    
    # save the buffer data as a txt file
    def save(self):
        with open(f"train_result_plot/buffer_data_{self.now}.txt", "w") as f:
            for state, action, reward, next_state, done, info in self.buffer:
                f.write(f"state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, done: {done}, info: {info}\n")
        print("Save the buffer data successfully!")
        


# SAC Agent
class SACAgent:
    def __init__(self, env, buffer_size=1000, batch_size=64, gamma=0.9, tau=0.005, alpha=0.1, lr=0.0003):
        self.env = env
        self.state_dim = env.observation_space.shape[0] + 2  # 增加2个维度以支持独热编码
        self.action_dim = env.action_space_tip.shape[0]
        self.action_bound = [env.action_space_tip.low, env.action_space_tip.high]

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Networks
        self.policy_net = MLP(self.state_dim, self.action_dim * 2)
        self.q_net1 = MLP(self.state_dim + self.action_dim, 1)
        self.q_net2 = MLP(self.state_dim + self.action_dim, 1)
        self.target_q_net1 = MLP(self.state_dim + self.action_dim, 1)
        self.target_q_net2 = MLP(self.state_dim + self.action_dim, 1)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)

        # Copy weights to target networks
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

    def one_hot_encode(self, state):
        """将分子状态（0, 1, 2）转换为独热编码，并保留反应时间"""
        molecule_state = int(state)  # 分子状态
        # reaction_time = state[1]  # 反应时间
        one_hot = np.zeros(3)
        one_hot[molecule_state] = 1  # 对分子状态进行独热编码
        return np.array(one_hot)

    def remap_action(self, action):
        """将 [-1, 1] 范围的动作重新映射到实际动作空间"""
        low, high = self.action_bound
        return (low + high)/ 2 + (high - low) * action / 2

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = torch.chunk(self.policy_net(state), 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        action = torch.tanh(action)  # 将动作限制在 [-1, 1]
        return self.remap_action(action.detach().numpy()[0])  # 重新映射到原始动作空间

    def train(self, num_steps, render=False, visualize=False):
        state = self.env.reset()
        state = self.one_hot_encode(state)  # 转换状态为独热编码并添加反应时间
        
        # plot the action
        if visualize:
            queue = multiprocessing.Queue(50)
            plot_process = multiprocessing.Process(target=self.plot_actions, args=(queue,self.action_bound))
            plot_process.start()
        
        for step in range(num_steps):
            action = np.array([0,0,self.select_action(state)[0],self.select_action(state)[1]])
            next_state, reward, done, info = self.env.step(action)
            print(f"Reward: {reward}")
            next_state = self.one_hot_encode(next_state)  # 转换为独热编码并添加反应时间
            self.replay_buffer.add(state, action, reward, next_state, done, info)

            if render:
                self.env.render()
                self.env.clock.tick(60)

            if visualize:
                
                if queue.full():  # 如果队列满了，则等待一段时间
                    queue.get()
                
                queue.put((state, action, reward, next_state, done, info))

            if self.replay_buffer.size() > self.batch_size:
                self.update()

            if step % 2000 == 0:
                self.save_model(step)

            if done:
                state = self.one_hot_encode(self.env.reset())  # 重置并编码
            else:
                state = next_state
            
        if visualize:
            queue.put(None)
            plot_process.join()
        
        self.replay_buffer.plot_train()
        self.replay_buffer.save()



    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_tip(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        next_actions = torch.FloatTensor(self.select_action(next_states))

        # Update Q-functions
        with torch.no_grad():
            q_target1 = self.target_q_net1(torch.cat([next_states, next_actions], dim=-1))
            q_target2 = self.target_q_net2(torch.cat([next_states, next_actions], dim=-1))
            q_target = rewards + (1 - dones) * self.gamma * torch.min(q_target1, q_target2)

        q1 = self.q_net1(torch.cat([states, actions], dim=-1))
        q2 = self.q_net2(torch.cat([states, actions], dim=-1))
        q_loss1 = ((q1 - q_target) ** 2).mean()
        q_loss2 = ((q2 - q_target) ** 2).mean()

        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()

        # Update policy
        new_actions = torch.FloatTensor(self.select_action(states))
        q1_new = self.q_net1(torch.cat([states, new_actions], dim=-1))
        policy_loss = (self.alpha * -torch.log(q1_new) - q1_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

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
    def plot_actions(self,queue,action_bound):
        """绘图进程：消费队列中的数据并更新图表"""
        # 初始化绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title("x, y Distribution")
    
        # set the range of x and y, according to the action_bound
        ax1.set_xlim(-100,100)
        ax1.set_ylim(-100,100)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_title("v, a Distribution")
        # set the range of v and a, according to the action_bound
        ax2.set_xlim(action_bound[0][0],action_bound[1][0])
        ax2.set_ylim(action_bound[0][0],action_bound[1][0])
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
            ax1.scatter(data_x, data_y, c=colors_xy, alpha=0.5)
            ax2.scatter(data_v, data_a, c=colors_va, alpha=0.5)
            plt.pause(0.01)  # 实时刷新

        # plt.ioff()
        # plt.show()

# Train the agent
if __name__ == "__main__":
    env = Env()
    agent = SACAgent(env)
    agent.train(100000,visualize=True)
