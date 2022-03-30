#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author ZSQ
# @date 2022/3/20
# @file PG.py
import gym
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Categorical
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
'''
https://blog.csdn.net/weixin_44564247/article/details/120490412

tnnd,你学啊，为什么学不出来，学了1000轮啥都没有 
'''


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            # https://blog.csdn.net/weixin_43914889/article/details/104505512
            # https://blog.csdn.net/sunyueqinghit/article/details/101113251
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        return self.net(state)


rewards = []
episodes = []


class Trainer:
    def __init__(self):
        self.net = Agent()
        self.env = gym.make('CartPole-v1')
        # 定义优化器
        self.opt = torch.optim.Adam(self.net.parameters())

    def call(self):
        # 采样

        for epoch in range(1000):
            state = self.env.reset()
            memory = []  # 存放得分函数和回报
            gamma = 0.9  # 折扣系数
            # 走完一个episode
            while True:
                # if epoch>300:
                # self.env.render()
                action, log_prob = self.action_select(state)
                state, reward, done, info = self.env.step(action)
                memory.append([log_prob, reward])
                if done:
                    break
            # 计算总回报
            G = 0
            GS = []  # 存放每个状态的总回报
            total_rewards = 0  # 所有回报的和
            for _, reward in memory[::-1]:
                G = reward + G * gamma
                GS.insert(0, G)
                total_rewards += reward
            print(total_rewards)
            episodes.append(epoch)
            rewards.append(total_rewards)
            # 数据标准化
            GS = torch.tensor(GS)
            eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0
            GS = (GS - GS.mean()) / (GS.std() + eps)

            # 计算损失
            # https: // zhuanlan.zhihu.com / p / 110881517
            loss = 0
            for G, (log_prob, _) in zip(GS, memory):
                loss += -G * log_prob

            self.opt.zero_grad()  # 梯度初始化为零
            loss.backward()  # 反向传播
            self.opt.step()  # 更新所有参数

    def action_select(self, state):
        state = torch.from_numpy(state).float()
        probs = self.net(state[None])
        # print(probs)
        prob = probs[0]  # tensor([[1., 1.]], grad_fn=<SoftmaxBackward>)
        # 关于 https://blog.csdn.net/ProQianXiao/article/details/102893824
        m = Categorical(prob)  # 探索机制同时按照动作概率进行采样
        action = m.sample()  # 根据采样选择动作
        # https://blog.csdn.net/greepex/article/details/103466087
        return action.item(), m.log_prob(action)


if __name__ == '__main__':
    train = Trainer()
    train.call()
    plt.plot(episodes, rewards)
    plt.show()
