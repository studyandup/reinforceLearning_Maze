import random
import numpy as np
import pandas as pd
import time
import gym
import matplotlib.pyplot as plt
env = gym.make("Taxi-v3")
env.render()
'''
创建Q表的前提是知道多少状态和动作的维度
open ai 提供了两个接口 env.action_space.n env.observation_space_n
'''
action_size = env.action_space.n  # 获取动作维度（一个状态下有几种动作选择） # env.action_space.n env.observation.n
print("Action size ", action_size)
state_size = env.observation_space.n  # 获取状态维度（一共多少种状态）
print("State size ", state_size)
# 　初始化Ｑ表
qtable = np.zeros((state_size, action_size))
# Q-Learning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 确定超参数
total_episodes = 1000  # 一共玩多少局游戏
total_test_episodes = 100  # 测试中一共走几步
max_steps = 99  # 每一局游戏最多走几步

learning_rate = 0.7  # 学习率
gamma = 0.9  # 未来奖励折扣率
# 探索相关参数
epsilon = 1.0  # 探索概率
max_epsilon = 1.0  # 一开始的探索概率
min_epsilon = 0.01  # 最低探索概率
decay_rate = 0.01  # 探索概率的指数衰减概率
rewards = []
episodes = []
'''
Qlearning 算法
'''
for episode in range(total_episodes):
    # 重置环境
    state = env.reset()
    step = 0
    done = False  # 每一局最多走99步
    total_rewards = 0
    # ３。 choose an action in the current world State
    for step in range(max_steps):
        # 生成0-1之间的随机数
        exp_exp_tradeoff = random.uniform(0, 1)
        # 如果这个数字大于探索概率（初始概率为1），则进行利用（选择最大的Q的动作）
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        # 否则，选择一个随机的动作进行探索
        else:
            action = env.action_space.sample()
        # 这个动作与环境进行交互后，获得奖励，环境变成新的状态
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        # 按照公式 Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # 更新Q表
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        # qtable[state, action] = qtable[state, action] + learning_rate * (
        # reward + gamma * (qtable[new_state, action]) - qtable[state, action])
        # 迭代环境状态
        state = new_state
        # 如果游戏结束，则跳出循环
        if done:
            break
        # 减小探索概率
        epsilon = min_epsilon+(max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    episodes.append(episode)
    rewards.append(total_rewards)

plt.plot(episodes,rewards)
plt.show()

# main
env.reset()

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        # time.sleep(1)
        env.render()
        # 测试中我们就不需要探索了，只要选择最优动作
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward

        if done:
            print("Score", total_rewards)
            break
        state = new_state
env.close()

print ("Score over time: " + str(sum(rewards)/total_test_episodes))
# plt.plot(eepisode,)
