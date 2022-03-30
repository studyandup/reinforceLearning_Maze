#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/9/24
# @file AC.py
# !/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/9/21
# @file policyG.py
# 策略梯度算法
# 2020.5.22
#
# cartpole 的state是一个4维向量，分别是位置，速度，杆子的角度，加速度；action是二维、离散，即向左/右推杆子
# 每一步的reward都是1  游戏的threshold是475
# 这个是MC方法
# https://codeleading.com/article/60223771901/ 另外一个版本 ，连续空间的高斯分布 待续
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

SEED = 543  # 设置种子，以保证复现性
GAMMA = 0.99  # 学习率
RENDER = False  # 是否渲染画面
LOG_INTERVAL = 10  # 每隔10轮在控制台输出相关信息
HIDDEN_SIZE = 32  # 中间层节点数目

eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0

env = gym.make('CartPole-v1')  # 创建环境
rewards = []
episodes = []
env.seed(SEED)
torch.manual_seed(SEED)  # 策略梯度算法方差很大，设置seed以保证复现性
print('observation space:', env.observation_space)
print('action space:', env.action_space)
'''
https://zhuanlan.zhihu.com/p/110998399
AC 半懂不懂
Actor在台上跳舞，一开始舞姿并不好看，Critic根据Actor的舞姿打分。
Actor通过Critic给出的分数，去学习：
如果Critic给的分数高，那么Actor会调整这个动作的输出概率；
相反，如果Critic给的分数低，那么就减少这个动作输出的概率。
'''

class Agent(nn.Module):
    ##  离散空间采用了 softmax policy 来参数化策略
    def __init__(self):
        super(Agent, self).__init__()
        '''
        dropout的作用是增加网络的泛化能力，可以用在卷积层和全连接层。但是在卷积层一般不用dropout
        https://blog.csdn.net/junbaba_/article/details/105673998
        '''
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        '''
        
        '''
        # 动作概率网络
        self.action_net = nn.Sequential(
            nn.Linear(32, 2),
            # nn.Softmax(dim=1),
        )
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(32, 1),
        )

    def forward(self, state):
        x = self.net(state)
        value = self.value_net(x)
        action_scores = self.action_net(x)
        action_scores=torch.unsqueeze(action_scores, 0)
        probs = F.softmax(action_scores, dim=1)
        return probs, value


class Trainer:
    def __init__(self):
        self.net = Agent()
        self.opt = optim.Adam(self.net.parameters(), lr=1e-2)

    def __call__(self):
        for epoch in range(1000):
            # 采样
            state = env.reset()
            memory = []  # 存放的分函数和回报
            while True:
                # if epoch > 300:
                    # env.render()
                action, log_prob, value = self.__action_select(state)
                state, reward, done, info = env.step(action)
                memory.append([log_prob, action, value])
                if done:
                    break
            # 总回报
            G = 0
            GS = []  # 存放每一个状态的的总回报
            total_rewards = 0
            for _, reward, _ in memory[::-1]:
                G = reward + GAMMA + G
                GS.insert(0, G)
                total_rewards += reward
            print(total_rewards)
            episodes.append(epoch)
            rewards.append(total_rewards)
            # 数据标准化
            GS = torch.tensor(GS)
            GS = (GS - GS.mean()) / (GS.std() + eps)

            # 计算损失
            actor_loss = 0
            critic_loss = 0
            for G, (log_prob, _, value) in zip(GS, memory):
                actor_loss += -(G - value) * log_prob
                critic_loss += (value - G) ** 2
            # ?
            loss = actor_loss + critic_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def __action_select(self, state):
        state = torch.from_numpy(state).float()
        probs, values = self.net(state)  # 传过来的数值是一个列表
        prob = probs[0]
        value = values[0]
        m = Categorical(prob)  # 探索机制，按照动作的概率进行采用
        action = m.sample()  # 根据采样选择动作
        return action.item(), m.log_prob(action), value


if __name__ == '__main__':
    train = Trainer()
    train()
    plt.plot(episodes, rewards)
    plt.show()
