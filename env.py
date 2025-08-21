import random
from math import cos, radians, sin, sqrt

import numpy as np
import pygame
from gym import spaces

from utils import *


class Molecule:
    def __init__(self, position=(0, 0), shape='triangle', size=100, state=0, angle=0):
        self.position = position
        self.shape = shape
        self.size = size
        self.state = state
        self.angle = angle % 360
        self.delta_vector = (0, 0)
        self._key_points = []  # For interaction (tip)
        self._render_points = []  # For origin shape
        self.key_points = []  # For interaction (tip)
        self.render_points = []  # For rendering shape

        # 根据形状设置 key_points 和 render_points
        if shape == 'circle':
            self._key_points = [(self.position[0], self.position[1])]  # center as key point
            self._render_points = [(self.position[0], self.position[1])]  # center for rendering
            self.key_points = self._update_points(self._key_points)  # Rotate key points (for tip interaction)
            self.render_points = self._update_points(self._render_points)  # Rotate render points (for drawing)
        
        elif shape == 'triangle':
            # interaction parameters
            self.key_points_size_factor = 0.6
            self.center_std_devs = 18
            self.center_weights = 1
            self.edge_std_devs = 15
            self.edge_weights = 0.6            
            # _key_points: center + 3 vertices (for interaction)
            self._key_points = [
                (self.position[0], self.position[1]),  # Center
                (self.position[0], self.position[1] - sqrt(3) * self.size / 3),  # top
                (self.position[0] - self.size / 2, self.position[1] + sqrt(3) * self.size / 6),  # left
                (self.position[0] + self.size / 2, self.position[1] + sqrt(3) * self.size / 6),  # right
            ]
            # _render_points: 3 vertices for triangle shape
            self._render_points = [
                (self.position[0], self.position[1] - sqrt(3) * self.size / 3),  # top
                (self.position[0] - self.size / 2, self.position[1] + sqrt(3) * self.size / 6),  # left
                (self.position[0] + self.size / 2, self.position[1] + sqrt(3) * self.size / 6),  # right
            ]
            self.key_points = self._update_points(self._key_points)  # Rotate key points (for tip interaction)
            self.render_points = self._update_points(self._render_points)  # Rotate render points (for drawing)   

            self.mean = ([0, 0], [0, 0], [0, 0], [0, 0])
            self.std_devs = ([self.edge_std_devs, self.edge_std_devs], [self.edge_std_devs, self.edge_std_devs], [self.edge_std_devs, self.edge_std_devs], [self.center_std_devs, self.center_std_devs])
            self.weights = [self.edge_weights, self.edge_weights, self.edge_weights, self.center_weights]

            self.optimal_bias = 4.0
            self.optimal_current = 0.2



        elif shape == 'square':
            # interaction parameters
            self.key_points_size_factor = 0.6
            self.center_std_devs = 18
            self.center_weights = 1
            self.edge_std_devs = 16
            self.edge_weights = 0.5 
            # _key_points: center + 4 vertices
            self.size = 200
            half_size = self.size / 2
            self.state = np.array([0, 0, 0, 0])
            self._key_points = [
                (self.position[0], self.position[1]),  # Center
                (self.position[0] + half_size, self.position[1] ),  # top left
                (self.position[0] , self.position[1] + half_size),  # bottom left
                (self.position[0] - half_size, self.position[1] ),  # bottom right
                (self.position[0] , self.position[1] - half_size),  # top right
            ]
            # _render_points: 4 vertices for square shape
            self._render_points = [
                (self.position[0] + half_size, self.position[1] ),  # top left
                (self.position[0] , self.position[1] + half_size),  # bottom left
                (self.position[0] - half_size, self.position[1] ),  # bottom right
                (self.position[0] , self.position[1] - half_size),  # top right
            ]
        
            self.key_points = self._update_points(self._key_points)  # Rotate key points (for tip interaction)
            self.render_points = self._update_points(self._render_points)  # Rotate render points (for drawing)

    def rotate(self, delta_angle):
        """
        Rotate all points (key_points and render_points) by delta_angle degrees.
        """
        self.angle = (self.angle + delta_angle) % 360
        self.key_points = self._update_points(self._key_points)  # Rotate key points (for tip interaction)
        self.render_points = self._update_points(self._render_points)  # Rotate render points (for drawing)

    def _update_points(self, points, scale_factor=1, translation_vector=(0, 0)):
        """
        Helper function to update points (rotate them).
        """
        # Convert angle from degrees to radians
        angle_rad = radians(self.angle)
        # Unpack the center of rotation and the translation vector
        cx, cy = self.position
        dx, dy = translation_vector

        transformed_points = []

        for (x, y) in points:
            # Step 1: Scale the point
            x_scaled = cx + scale_factor * (x - cx)
            y_scaled = cy + scale_factor * (y - cy)

            # Step 2: Rotate the scaled point
            x_rot = (x_scaled - cx) * cos(angle_rad) - (y_scaled - cy) * sin(angle_rad) + cx
            y_rot = (x_scaled - cx) * sin(angle_rad) + (y_scaled - cy) * cos(angle_rad) + cy

            # Step 3: Translate the rotated point
            x_translated = x_rot + dx
            y_translated = y_rot + dy

            # Append the transformed point to the result list
            transformed_points.append((x_translated, y_translated))
        return transformed_points


    def move(self, translation_vector=(0, 0)):
        """
        Move the molecule by dx and dy.
        """
        self.delta_vector = self.delta_vector + translation_vector
        self.position = (self.position[0] + translation_vector[0], self.position[1] + translation_vector[1])
        self.key_points = self._update_points(self._key_points, translation_vector=self.delta_vector)
        self.render_points = self._update_points(self._render_points, translation_vector=self.delta_vector)

    def interact_area(self, interact_position):
        if self.shape == 'triangle':
            centers = self.key_points
            means = ([0, 0], [0, 0], [0, 0], [0, 0])
            std_devs = ([self.edge_std_devs, self.edge_std_devs], [self.edge_std_devs, self.edge_std_devs], [self.edge_std_devs, self.edge_std_devs], [self.center_std_devs, self.center_std_devs])
            weights = [self.edge_weights, self.edge_weights, self.edge_weights, self.center_weights]
            
            x = np.linspace(self.position[0]-self.size, self.position[1]+self.size, self.size)
            y = x
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            value = 0
            for center, mean, std_dev, weight in zip(centers, means, std_devs, weights):
                x0, y0 = center
                mean_x, mean_y = mean
                std_x, std_y = std_dev
                
                # 计算二维正态分布
                Z += (1 / (2 * np.pi * std_x * std_y)) * np.exp(
                    -(((X - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
                    ((Y - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
                )
            
                value += (1 / (2 * np.pi * std_x * std_y)) * np.exp(
                    -(((interact_position[0] - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
                    ((interact_position[1] - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
                )
            # Z = Z / np.max(Z)
            value = value / np.max(Z)

            return value

        elif self.shape == 'square':
            return 0
    
    def interact_parameters(self, interact_bias, interact_current):
        if self.shape == 'triangle':

            delta_bias = interact_bias - self.optimal_bias
            delta_current = interact_current - self.optimal_current
            scale_factor_bias = 3
            scale_factor_current = 12
            x = delta_bias*scale_factor_bias
            y = delta_current*scale_factor_current
            suc_rate = suc(x, y)
            ind_rate = ind(x, y)
            non_rate = 1-suc_rate-ind_rate
            return suc_rate, ind_rate, non_rate

        if self.shape == 'square':
            return 0, 0, 0

    def change_state(self, new_state):
        self.state = new_state

class Surface:
    def __init__(self, size=(500, 500), angle=0):
        self.size = size
        self.angle = angle
        self.molecules = []
        self.molecules_distance = 138 # 138 → 1.38nm
        self.grid_size = (1, 1)
        self.center = (self.size[0] / 2, self.size[1] / 2)
        self._position_list = generate_triangle_grid(self.center, self.grid_size, self.molecules_distance)
        
        self.position_list = []

        self.position_list = self._update_points(self._position_list)

    def adsorb(self, position, shape='triangle', size=100, state=0, angle=0):      #size=100 → 1nm
        """
        Creates and absorbs a new molecule.
        """
        molecule = Molecule(position=position, shape=shape, size=size, state=state, angle=angle)
        self.molecules.append(molecule)
    
    def _update_points(self, points, scale_factor=1, translation_vector=(0, 0)):
        """
        Helper function to update points (rotate them).
        """
        # Convert angle from degrees to radians
        angle_rad = radians(self.angle)
        # Unpack the center of rotation and the translation vector
        cx, cy = self.center
        dx, dy = translation_vector

        transformed_points = []

        for (x, y) in points:
            # Step 1: Scale the point
            x_scaled = cx + scale_factor * (x - cx)
            y_scaled = cy + scale_factor * (y - cy)

            # Step 2: Rotate the scaled point
            x_rot = (x_scaled - cx) * cos(angle_rad) - (y_scaled - cy) * sin(angle_rad) + cx
            y_rot = (x_scaled - cx) * sin(angle_rad) + (y_scaled - cy) * cos(angle_rad) + cy

            # Step 3: Translate the rotated point
            x_translated = x_rot + dx
            y_translated = y_rot + dy

            # Append the transformed point to the result list
            transformed_points.append((x_translated, y_translated))
        return transformed_points
    
    # def a function to rotate whole surface
    def rotate(self, delta_angle):
        """
        Rotate all points (key_points and render_points) by delta_angle degrees.
        """
        self.angle = (self.angle + delta_angle) % 360
        self.position_list = self._update_points(self._position_list)

    def arrange_molecules(self):
        
        for position in self.position_list:
            self.adsorb(position, shape='triangle', size=200)
        self.position_list = self._update_points(self._position_list)
        

class Tip:
    def __init__(self, position=(0, 0)):
        self.position = position
        self.reaction_time = 0
        self.bias_voltage = -4.1
        self.current = -0.11
        self.suc_factor = 0
        self.fail_factor = 0
        self.non_factor = 0
        
    def move(self, new_position):
        self.position = new_position
    
    def change_bias(self, delta_bias):
        self.bias_voltage += delta_bias

    def change_current(self, delta_current):
        self.current += delta_current

    def culculate_reaction_factor(self, molecule, weight_suc = 0.9, weight_fail = 0.05):

        position_factor = molecule.interact_area(self.position)

        parameters_suc_factor, parameters_ind_factor, parameters_non_factor = molecule.interact_parameters(self.bias_voltage, self.current)
        # print("parameters_ind_factor:", parameters_ind_factor)

        suc_factor = position_factor*parameters_suc_factor*weight_suc
        fail_factor = position_factor*weight_fail + parameters_ind_factor
        position_non_factor = 1 - suc_factor - fail_factor
        
        self.suc_factor = suc_factor
        self.fail_factor = fail_factor
        self.non_factor = position_non_factor

        return suc_factor, fail_factor, position_non_factor

    def interact(self, molecule):

        bias_voltage = self.bias_voltage
        current = self.current

        distance = np.linalg.norm(np.array(self.position) - np.array(molecule.position))
        # print('distance:', distance)
        
        if distance > molecule.size*1.5:
            self.reaction_time = 10
            return 0
        self.bias_voltage = bias_voltage
        self.current = current

        # self.culculate_reaction_factor(molecule)
        #
        # position_weight_suc is 0.8 and gussian noise ± 0.1
        # weight_suc = 0.9 + random.gauss(0, 0.1)
        weight_suc = 0.9
        # position_weight_fail is 0.05 and gussian noise ± 0.05
        # weight_fail = 0.05 + random.gauss(0, 0.05)
        weight_fail = 0.05
        # the 
        position_factor = molecule.interact_area(self.position)

        parameters_suc_factor, parameters_ind_factor, parameters_non_factor = molecule.interact_parameters(self.bias_voltage, self.current)
        # print("parameters_ind_factor:", parameters_ind_factor)

        suc_factor = position_factor*parameters_suc_factor*weight_suc
        fail_factor = position_factor*weight_fail + parameters_ind_factor
        position_non_factor = 1 - suc_factor - fail_factor
        
        self.suc_factor = suc_factor
        self.fail_factor = fail_factor
        self.non_factor = position_non_factor
        #calculate the reaction time, the reaction is position_non_factor*parameters_non_factor*10 but it should be between 0 and 8
        # self.reaction_time = max(min((position_non_factor+parameters_non_factor)*6 + random.gauss(0, 1), 8), 0.01)

        self.reaction_time = max((position_non_factor+parameters_non_factor)*6 + random.gauss(0, 1), 0.01)
        # print('suc_factor:', suc_factor)
        # print('fail_factor:', fail_factor)
        # print('non_factor:', position_non_factor)
        # print('reaction_time:', self.reaction_time)
        if molecule.state == 0:
            action_random = random.random()
            # print(position_suc_factor)

            if action_random < suc_factor:
                molecule.change_state(1)

            elif action_random > 1 - fail_factor:
                molecule.change_state(2)
            else:
                self.reaction_time = random.choice([self.reaction_time, 8])


        # success_probability = max(0.8 - 0.05 * bias_deviation - 0.05 * current_deviation - 0.1 * distance_penalty, 0)
        # fail_probability = max(0.5 + 0.1 * bias_deviation + 0.1 * current_deviation - 0.1 * distance_penalty, 0)

class Env:
    def __init__(self,polar_space = False):
        self.surface = Surface()
        self.tip = Tip()
        # self.screen = pygame.display.set_mode((1000,1000))
        self.running = True
        self.polar_space = polar_space
        self.render_flag = 1

        if self.polar_space:
            action_low_x, action_high_x = 0, 70
            action_low_y, action_high_y = 0, 360
        else:
            action_low_x, action_high_x = -55,55
            action_low_y, action_high_y = -55, 55


        self.action_space = spaces.Box(
            low=np.array([action_low_x, action_low_y, -5.0, 0.02]),       #low bias, low current
            high=np.array([action_high_x, action_high_y, -3.5, 0.7]),    #high bias, high current
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([0]),
            high=np.array([2]),
            dtype=np.float32,
        )

        self.action_space_tip = spaces.Box(
            low=np.array([ 2, 0.1]),
            high=np.array([ 4, 1.5]),
            dtype=np.float32,
        )

    def render(self):
        # if self.render_flag == 1:# initialize the pygame
        #     pygame.init()
        #     self.screen = pygame.display.set_mode(self.surface.size, pygame.RESIZABLE)
        #     pygame.display.set_caption("Interactive STM Environment")
        #     self.clock = pygame.time.Clock()
        #     self.render_flag = 0  # set the flag to 0 to avoid initialize the pygame again
        self.screen.fill((255, 255, 255))
        for molecule in self.surface.molecules:
            if molecule.shape == 'triangle':
                color = (0, 0, 255) if molecule.state == 0 else (0, 255, 0) if molecule.state == 1 else (255, 0, 0)
                pygame.draw.polygon(self.screen, color, molecule.render_points)
                # draw the key points
                for point in molecule.key_points:
                    pygame.draw.circle(self.screen, (0, 0, 0), (int(point[0]), int(point[1])), 3)
            elif molecule.shape == 'square':
                body_color = (0, 0, 255) 
                sub_body_color = (0, 0, 128)
                pygame.draw.polygon(self.screen, body_color, molecule.render_points)
                for point in molecule.render_points:
                    pygame.draw.circle(self.screen, sub_body_color, (int(point[0]), int(point[1])), 30)
                # draw the key points
                for point in molecule.key_points:
                    pygame.draw.circle(self.screen, (0, 0, 0), (int(point[0]), int(point[1])), 3)


        pygame.draw.circle(self.screen, (0, 255, 0), self.tip.position, 5)
        # show the bias voltage and current at the top left corner
        font = pygame.font.SysFont("arial", 12)
        text = font.render("Bias Voltage: {:.2f}V  Current: {:.2f}nA".format(self.tip.bias_voltage, self.tip.current), True, (0, 0, 0))
        self.screen.blit(text, (10, 20))
        # show the success rate, indeterminate rate and failure rate at the bottom left corner
        text_1 = font.render("Suc: {:.2f}  Non: {:.2f}  Fail: {:.2f}".format(self.tip.suc_factor, self.tip.non_factor, self.tip.fail_factor), True, (0, 0, 0))
        self.screen.blit(text_1, (10, 480))
        pygame.display.flip()

    def reset(self):
        """
        重置环境到初始状态，并返回初始观测值。
        """
        self.surface.molecules = []
        self.surface.arrange_molecules()
        for molecule in self.surface.molecules:
            molecule.change_state(0)
        self.tip.move((self.surface.center[0], self.surface.center[1]))
        molecule = self.surface.molecules[0]  # 默认选择第一个分子
        # self.current_molecule = molecule
        return np.array(molecule.state)
    
    # culculate the reward
    def reward_culculate(self,state_init, state, reaction_time):
        if state_init==0 and state == 1:                # success
            info ={"reaction" : "success", "reaction_time" : reaction_time}
            return 15 - 0.2 * reaction_time, info
            
        elif state_init==0 and state == 0:              # none
            info ={"reaction" : "none", "reaction_time" : reaction_time}
            return -0.1 - 0.2 * reaction_time, info
        
        elif state_init==0 and state == 2:              # failure
            info ={"reaction" : "failure", "reaction_time" : reaction_time}
            return -5 - 0.2 * reaction_time, info
        else:
            return -10, {"reaction" : "NO STATE ERROR!"}

    def step(self, action):
        """
        执行动作，更新环境状态并返回:
        - 观测值
        - 奖励
        - 是否结束
        - 附加信息
        """
        if self.polar_space:
            l, theta, v, a = action # theta in 0-360
            x = l * np.cos(np.radians(theta))
            y = l * np.sin(np.radians(theta))
        else:
            x, y, v, a = action
        # self.tip.move((x, y))
        for molecule in self.surface.molecules[0:]:
            state_init = molecule.state
            self.tip.move((molecule.position[0]+x, molecule.position[1]+y))
            self.tip.bias_voltage = -v
            self.tip.current = a
            self.tip.interact(molecule)

        # 获取反应结果
        state = molecule.state
        reaction_time = self.tip.reaction_time
        reaction_time = 0

        reward, info = self.reward_culculate(state_init, state, reaction_time)

        # 是否结束
        done = state != 0

        # 返回新的状态和奖励
        return np.array(state, dtype=np.float32), reward, done, info
    
    def run(self):
        self.surface.arrange_molecules()
        # initialize the pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.surface.size, pygame.RESIZABLE)
        pygame.display.set_caption("Interactive STM Environment")
        self.clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEMOTION:
                    self.tip.move(event.pos)
                    for molecule in self.surface.molecules:
                        self.tip.culculate_reaction_factor(molecule)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for molecule in self.surface.molecules:
                        self.tip.interact(molecule)
                # if hit the 'R' key, reset the all melocules to state 0
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    for molecule in self.surface.molecules:
                        molecule.change_state(0)
                # if hit the 'space' key, print the position of the all molecules
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    for molecule in self.surface.molecules:
                        # print(molecule.key_points)
                        print(molecule.render_points)
                # if hit the 'right' key, rotate the all molecules 1 degree clockwise, 'left' key for counter-clockwise
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    for molecule in self.surface.molecules:
                        molecule.rotate(1)
                    print(molecule.angle)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    for molecule in self.surface.molecules:
                        molecule.rotate(-1)
                    print(molecule.angle)
                # if hit the 'w' key, up the tip add 0.05 to the tip bias voltage, 's' key for down
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                    self.tip.change_bias(0.05)
                    print("bias:",self.tip.bias_voltage)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.tip.change_bias(-0.05)
                    print("bias:",self.tip.bias_voltage)
                # if hit the 'a' key, up the tip add 0.05 to the tip current, 'd' key for down
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    self.tip.change_current(0.05)
                    print("current:",self.tip.current)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                    self.tip.change_current(-0.05)
                    print("current:",self.tip.current)
            self.render()
            self.clock.tick(60)
        pygame.quit()

    def close(self):
        """
        关闭环境。
        """
        self.running = False
        pygame.quit()

if __name__ == "__main__":

    env = Env()
    env.run()
