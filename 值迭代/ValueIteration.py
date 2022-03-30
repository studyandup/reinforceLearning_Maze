#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author ZSQ
# @date 2022/3/29
# @file ValueIteration.py

from Env import Maze
import numpy as np
'''
明天： 
1。 继续完成价值迭代代码， 策略迭代代码，异步价值迭代，异步策略迭代
2.  完成TD（lambda)
'''

# 状态转移概率
prob = 1.0
def ValueIteration(env,theta = 0.01,discount_factor=0.8):
    def one_step_action_choice(s,value):
        env.state = s
        #　初始化该状态该节点的value
        temp_value = np.zeros(len(env.actions))
        # 采取不同的动作，计算该状态下该节点不同动作下的value值
        for a in env.actions:
            # 这里不应该使用step，若使用则改变s状态, 在下面加了env.state =s 重置状态为 s
            next_state, reward, is_terminal, info = env.step(a)
            # 此处的a为 ['up', 'right', 'down', 'left'] 不是数字
            temp_value[env.actions_ky[a]] += prob * (reward + discount_factor * value[next_state])
            env.state = s
        #     if s ==9:
        #         print("s:{}  a:{}  reward:{}, next_state:{} ".format(s, a, reward,next_state))
        # if s==9:
        #     print("9号节点的value值： {}".format(temp_value))
        return temp_value

     #　初始化value值
    value = np.zeros(len(env.states))
    # 迭代
    '''
    注意这里，我们的迷宫是有墙壁的，而例子没有，注意修改
    '''
    step_iteraton = 0
    while True:
        delta = 0
        for s in env.safeStates:
            temp_value = one_step_action_choice(s,value)
            best_action_value = np.max(temp_value)
            # Calculate terminate condition
            delta = max(delta, np.abs(best_action_value - value[s]))
            # 更新value
            value[s] = best_action_value
        print("第"+ str(step_iteraton) +"轮次迭代： ")
        for i in range(25):
             print("{}: {}".format(i,value[i]),end=" ")
        print("-- ")
        step_iteraton += 1
        # check if we can stop
        if delta < theta:
            break
    policy = np.zeros([len(env.states),len(env.actions)])
    for s in env.safeStates:
        temp_value = one_step_action_choice(s,value)
        best_action = np.argmax(temp_value)
        policy[s,best_action] = 1.0
    return policy, value

if __name__ == '__main__':
    env = Maze()
    policy, V = ValueIteration(env)
    print("Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(V)
    print(np.argmax(policy, axis=1))
    # 直观化策略
    action_ky = {0:'up',1:'right',2:'down',3:'left'}
    temp = np.argmax(policy, axis=1)
    for i in temp:
        if i in env.safeStates:
            print("{}号节点策略：{}".format(i,action_ky[temp[i]]))
