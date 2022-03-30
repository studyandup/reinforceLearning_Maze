#!/user/bin/env python3
# -*- conding:utf-8 -*-
# @auther Zhang
# @date 2021/9/11
# @file DQN.py
# -*- coding: utf-8 -*-
# import the necessary packages
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
''' 
DQN--------------------------------------------------------------------
'''
# 1. 定义超参数
BATCH_SIZE = 32  # batch size of sampling process from buffer
LR = 0.01  # 学习率
EPSILON = 0.9  # epsilon used for epsilon greedy approach
GAMMA = 0.9  # 衰减值
TARGET_NETWORK_REPLACE_FREQ = 100  # How frequently target netowrk updates
MEMORY_CAPACITY = 2000  # The capacity of experience replay buffer

env = gym.make("CartPole-v0")  # Use cartpole game as environment
# env = env.unwrapped
N_ACTIONS = env.action_space.n  # 2 actions
N_STATES = env.observation_space.shape[0]  # 4 states
# to confirm the shape
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),int) else env.action_space.sample().shape



# 2. Define the network used in both target net and the net for training
class Net(nn.Module):
    def __init__(self):
        # Define the network structure, a very simple fully connected network
        super(Net, self).__init__()
        # Define the structure of fully connected network 全连接网络
        # https://blog.csdn.net/m0_37586991/article/details/87861418
        # 此处设置了两个全连接层，随机权重
        self.fc1 = nn.Linear(N_STATES, 100)  # layer 1
        # 将tensor用均值为0和标准差为0.1的正态分布填充。
        self.fc1.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc1

        self.fc2 = nn.Linear(100,100)
        self.fc2.weight.data.normal_(0,0.1)

        self.out = nn.Linear(100, N_ACTIONS)  # layer 2
        self.out.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc2

    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# 3. Define the DQN network and its corresponding methods
class DQN(object):
    def __init__(self):
        # -----------Define 2 networks (target and training)------#
        # 定义目标网络和评估网络
        self.eval_net, self.target_net = Net(), Net()
        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # ----Define the memory (or the buffer), allocate some space to it. The number
        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        # s
        # Num  Observation                Min                      Max
        # 0     Cart Position             -4.8                    4.8
        # 1     Cart Velocity             -Inf                    Inf
        # 2     Pole Angle                -0.418 rad(-24 deg      0.418 rad (24 deg)
        # 3     Pole Angular Velocity     -Inf                    Inf
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

        # ------- 定义优化器------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        # ------定义均方损失函数-----#
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy
        # ？？？？？？？？？？？？？？？？？？？  np.stack()  torch.unsqueeze()
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # add 1 dimension to input state x
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            # print(torch.max(actions_value, 1))
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            # torch.max(input, dim) 函数
            # input是softmax函数输出的一个tensor
            # dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            # numpy.random.randint(low, high=None, size=None, dtype='l')
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer
        # np.hstack():在水平方向上平铺 np.vstack():水平(按列顺序)把数组给堆叠起来
        transition = np.hstack((s, [a, r], s_))  # horizontally stack these vectors
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps 运行既定步数后更新目标网络
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        # numpy.random.choice(a, size=None, replace=True, p=None)
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        # Variable是对Tensor的一个封装，操作和Tensor是一样的，
        # 但是每个Variable都有三个属性，Varibale的Tensor本身的.data，对应Tensor的梯度.grad，
        # 以及这个Variable是通过什么方式得到的.grad_fn。
        # torch.FloatTensor 类型转换, 将list ,numpy转化为tensor
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # calculate the Q value of state-action pair
        # gather() https://blog.csdn.net/edogawachia/article/details/80515038
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # q_eval = self.target_net(b_s).gather(1,b_a)
        # print(q_eval)
        # calculate the q value of next state
        # detach() 返回一个tensor变量，且这个变量永远不会有梯度值。这个变量跟原图上的变量共享一块内存，也就说是同一个家伙。
        q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        # 梯度初始化为零
        self.optimizer.zero_grad()  # reset the gradient to zero
        # 反向传播求梯度
        loss.backward()
        # 更新所有参数
        self.optimizer.step()  # execute back propagation for one step


'''
--------------Procedures of DQN Algorithm------------------
'''
# create the object of DQN class
dqn = DQN()
episode = []
epr = []
# Start training
print("\nCollecting experience...")
for i_episode in range(400):
    # play 400 episodes of cartpole game
    s = env.reset()
    ep_r = 0
    while True:
        # env.render()
        # take action based on the current state
        # actions_value = self.eval_net.forward(x)
        a = dqn.choose_action(s)
        # obtain the reward and next state and some other information
        s_, r, done, info = env.step(a)

        # modify the reward based on the environment state ??????????
        # s
        # Num  Observation                Min                      Max
        # 0     Cart Position             -4.8                    4.8
        # 1     Cart Velocity             -Inf                    Inf
        # 2     Pole Angle                -0.418 rad(-24 deg      0.418 rad (24 deg)
        # 3     Pole Angular Velocity     -Inf                    Inf
        #
        Cart_Position, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity  = s_
        r1 = (env.x_threshold - abs(Cart_Position)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(Pole_Angle)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # store the transitions of states
        dqn.store_transition(s, a, r, s_)

        ep_r += r
        # if the experience repaly buffer is filled, DQN begins to learn or update
        # its parameters.
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                epr.append(round(ep_r, 2))
                episode.append(i_episode)
                print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

        if done:
            # if game is over, then skip the while loop.
            break
        # use next state to update the current state.
        s = s_
env.close()

plt.plot(episode, epr)
plt.xlabel("episode")
plt.ylabel("ep_reward")
plt.show()