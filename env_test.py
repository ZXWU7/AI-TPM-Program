from gym import spaces
import numpy as np
import pygame

class Env:
    def __init__(self):
        pygame.init()
        self.surface = Surface()
        self.tip = Tip()
        self.screen = pygame.display.set_mode(self.surface.size, pygame.RESIZABLE)
        pygame.display.set_caption("Interactive STM Environment")
        self.clock = pygame.time.Clock()
        self.running = True

        # 定义动作空间: [X, Y, V, A]
        self.action_space = spaces.Box(
            low=np.array([-200, -200, 3, 0.3]),
            high=np.array([200, 200, 4, 1.7]),
            dtype=np.float32,
        )

        # 定义观测空间: [分子状态, 反应时间]
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([2, 8]),
            dtype=np.float32,
        )

    def reset(self):
        """
        重置环境到初始状态，并返回初始观测值。
        """
        self.surface.arrange_molecules()
        for molecule in self.surface.molecules:
            molecule.change_state(0)
        self.tip.move((self.surface.center[0], self.surface.center[1]))
        molecule = self.surface.molecules[0]  # 默认选择第一个分子
        self.current_molecule = molecule
        return np.array([molecule.state, 0], dtype=np.float32)

    def step(self, action):
        """
        执行动作，更新环境状态并返回:
        - 观测值
        - 奖励
        - 是否结束
        - 附加信息
        """
        x, y, v, a = action
        self.tip.move((x, y))
        self.tip.interact(self.current_molecule, bias_voltage=v, current=a)

        # 获取反应结果
        state = self.current_molecule.state
        reaction_time = self.tip.reacrion_time

        # 奖励函数
        if state == 1:  # 成功反应
            reward = 1 - 0.2 * reaction_time
        elif state == 0:  # 没反应
            reward = -0.2 * reaction_time
        elif state == 2:  # 分子被破坏
            reward = -5 - 0.2 * reaction_time
        else:
            reward = -10  # 未知错误的惩罚

        # 是否结束
        done = state != 0

        # 返回新的状态和奖励
        return np.array([state, reaction_time], dtype=np.float32), reward, done, {}

    def render(self):
        """
        渲染环境。
        """
        self.screen.fill((255, 255, 255))
        for molecule in self.surface.molecules:
            color = (0, 0, 255) if molecule.state == 0 else (0, 255, 0) if molecule.state == 1 else (255, 0, 0)
            pygame.draw.polygon(self.screen, color, molecule.render_points)
        pygame.draw.circle(self.screen, (0, 255, 0), self.tip.position, 5)
        pygame.display.flip()

    def close(self):
        """
        关闭环境。
        """
        pygame.quit()
