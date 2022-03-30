import gym
from numpy import random
import time


class Maze(gym.Env):
    def __init__(self):
        self.viewer = None
        # 状态空间 大小为 26
        self.states = [0,  1,  2,  3, 4,
                       5,  6,  7,  8, 9,
                       10,11, 12, 13, 14,
                       15,16, 17, 18, 19,
                       20,21, 22, 23, 24]
        # 长度为 19
        self.safeStates = [0,1,2,4,5,6,7,9,12,13,14,15,16,17,18,19,20,21]
        # 动作空间 大小为4
        self.actions = ['up', 'right', 'down', 'left']
        self.actions_ky = {'up':0,'right':1,'down':2,'left':3}
        # 0:up  1:right  2:down  3:left
        # self.actions = [0,1,2,3]
        # 回报函数
        self.rewards = dict()
        self.rewards['9_down'] = 10
        self.rewards['13_right'] = 10
        self.rewards['19_up'] = 10

        # 状态转移概率
        self.t = dict()
        self.t['0_down'] = 5
        self.t['0_right'] = 1
        self.t['1_left'] = 0
        self.t['1_down'] = 6
        self.t['1_right'] = 2
        self.t['2_left'] = 1
        self.t['2_down'] = 7
        self.t['4_down'] = 9
        self.t['5_up'] = 0
        self.t['5_right'] = 6
        self.t['6_left'] = 5
        self.t['6_up'] = 1
        self.t['6_right'] = 7
        self.t['7_left'] = 6
        self.t['7_up'] = 2
        self.t['7_down'] = 12
        self.t['9_up'] = 4
        self.t['9_down'] = 14
        self.t['12_up'] = 7
        self.t['12_right'] = 13
        self.t['12_down'] = 17
        self.t['13_left'] = 12
        self.t['13_right'] = 14
        self.t['13_down'] = 18
        self.t['15_right'] = 16
        self.t['15_down'] = 20
        self.t['16_left'] = 15
        self.t['16_right'] = 17
        self.t['16_down'] = 21
        self.t['17_left'] = 16
        self.t['17_right'] = 18
        self.t['17_up'] = 12
        self.t['18_left'] = 17
        self.t['18_up'] = 13
        self.t['18_right'] = 19
        self.t['19_left'] = 18
        self.t['19_up'] = 14
        self.t['20_up'] = 15
        self.t['20_right'] = 21
        self.t['21_left'] = 20
        self.t['21_up'] = 16

    def step(self, action):
        # state为当前状态
        state = self.state
        key = "%d_%s" % (state, action)
        # 选择是否输出动作
        # print(key)
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state
        is_terminal = False
        # 设定奖励 走到目标奖励为10，非目标奖励为-1。
        if key in self.rewards:
            r = 10
            is_terminal = True
        else:
            r = -1
        return next_state, r, is_terminal, {}

    def reset(self):
        # 初始位置不可使用的地点， 即是 墙
        s = [3, 8, 10, 11, 22, 23, 24,14]
        self.state = self.states[int(random.random() * (len(self.states) - 1))]
        while self.state in s:
            self.state = self.states[int(random.random() * (len(self.states) - 1))]
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering
        width = 60
        height = 40
        edge_x = 120
        edge_y = 50
        if self.viewer is None:
            self.viewer = rendering.Viewer(600, 400)

        # 右下角                 用黑色表示墙
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 2, edge_y + height * 1)))
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 3, edge_y + height * 1)))
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 4, edge_y + height * 1)))
        # 左边
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(rendering.Transform((edge_x, edge_y + height * 3)))
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 1, edge_y + height * 3)))
        # 上边
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 3, edge_y + height * 4)))
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(0, 0, 0)).add_attr(
            rendering.Transform((edge_x + width * 3, edge_y + height * 5)))
        # 出口，用黄色表示出口
        self.viewer.draw_polygon([(0, 0), (0, height), (width, height), (width, 0)], filled=True,
                                 color=(1, 0.9, 0)).add_attr(
            rendering.Transform((edge_x + width * 4, edge_y + height * 3)))
        # 画网格
        for i in range(1, 7):
            self.viewer.draw_line((edge_x, edge_y + height * i), (edge_x + 5 * width, edge_y + height * i))  # 横线
            self.viewer.draw_line((edge_x + width * (i - 1), edge_y + height),
                                  (edge_x + width * (i - 1), edge_y + height * 6))  # 竖线

        # 人的像素位置
        self.x = [edge_x + width * 0.5, edge_x + width * 1.5, edge_x + width * 2.5, 0, edge_x + width * 4.5,
                  edge_x + width * 0.5, edge_x + width * 1.5, edge_x + width * 2.5, 0, edge_x + width * 4.5,
                  0, 0, edge_x + width * 2.5, edge_x + width * 3.5, edge_x + width * 4.5,
                  edge_x + width * 0.5, edge_x + width * 1.5, edge_x + width * 2.5, edge_x + width * 3.5,
                  edge_x + width * 4.5,
                  edge_x + width * 0.5, edge_x + width * 1.5, 0, 0, 0]

        self.y = [edge_y + height * 5.5, edge_y + height * 5.5, edge_y + height * 5.5, 0, edge_y + height * 5.5,
                  edge_y + height * 4.5, edge_y + height * 4.5, edge_y + height * 4.5, 0, edge_y + height * 4.5,
                  0, 0, edge_y + height * 3.5, edge_y + height * 3.5, edge_y + height * 3.5,
                  edge_y + height * 2.5, edge_y + height * 2.5, edge_y + height * 2.5, edge_y + height * 2.5,
                  edge_y + height * 2.5,
                  edge_y + height * 1.5, edge_y + height * 1.5, 0, 0, 0]
        # 用圆表示人
        # self.viewer.draw_circle(18,color=(0.8,0.6,0.4)).add_attr(rendering.Transform(translation=(edge_x+width/2,edge_y+height*1.5)))
        self.viewer.draw_circle(18, color=(0.8, 0.6, 0.4)).add_attr(
            rendering.Transform(translation=(self.x[self.state ], self.y[self.state ])))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


# 环境测试
if __name__ == "__main__":
    env = Maze()
    env.reset()
    reward = 0
    step = 0

    while True:
        # 使用随机策略
        action = env.actions[int(random.random() * len(env.actions))]
        next_state, r, is_terminal, info = env.step(action)
        env.render()
        time.sleep(0.5)
        reward += r
        step += 1
        if is_terminal == True:
            print("reward:{},step:{}".format(reward,step))
            break
        # time.sleep(0.5)
    env.close()
