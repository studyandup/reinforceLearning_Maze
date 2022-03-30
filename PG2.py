#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author ZSQ
# @date 2022/3/24
# @file PG2.py
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

env = gym.make('CartPole-v1')
env.seed(1)
torch.manual_seed(1)

# 超参数
input_size = 4
hidden_size = 128
output_size = 2

lr = 0.01
gamma = 0.8


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.6)  # dropout 随机失活，减少过拟合

        # 储存
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.softmax(x, dim=1)   # 转换成概率，使其和为1
        return out


policy = Policy()
# policy.load_state_dict(torch.load('save_model.pt'))  # 模型导入
optimizer = optim.Adam(policy.parameters(), lr)
eps = np.finfo(np.float64).eps.item()
rewards = []
episodes = []

def choose_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def learn():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:   # 逆序遍历
        R = r + gamma * R
        returns.insert(0, R)   # 从头部插入
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)  # 归一化（均值方差），eps是一个非常小的数，避免除数为0

    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    # 反向传播，更新参数
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]  # 清空数据
    del policy.saved_log_probs[:]


def train_the_net(episode_num, time_to_render, time_to_save=470):
    RENDER = False
    Average_r = 100
    for i_episode in range(1, episode_num+1):
        s = env.reset()
        ep_r = 0

        for t in range(1, 1000):
            # if RENDER:
            #      env.render()  # 训练过程是否开启渲染

            a = choose_action(s)
            s, r, done, info = env.step(a)

            policy.rewards.append(r)
            ep_r += r

            if done:
                break

        Average_r = 0.05 * ep_r + (1 - 0.05) * Average_r
        learn()

        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(i_episode, ep_r, Average_r))

        if Average_r >= time_to_render:  # 什么时候开启渲染
            RENDER = True
            if Average_r >= time_to_save:
                torch.save(policy.state_dict(), 'save_model.pt')
                print("退出训练，保存模型")
                break


if __name__ == '__main__':
    train_the_net(1000, 200, 300)