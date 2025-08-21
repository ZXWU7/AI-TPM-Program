import os
import random
import time
from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, io

from core import NanonisController


class HindsightExperienceReplayBuffer:
    def __init__(self, max_size, env, her_k=4, goal_key='reaction'):
        """
        初始化 ReplayBuffer。

        参数：
        - max_size: 缓冲区的最大容量。
        - env: 环境对象，用于访问状态和奖励函数。
        - her_k: 每个回合中生成的 HER 替代经验的数量。
        - goal_key: 在状态或 info 字典中表示目标的键。
        """
        self.buffer = deque(maxlen=max_size)
        self.env = env
        self.her_k = her_k  # 每个回合中进行 HER 的次数
        self.goal_key = goal_key  # 目标在 state 或 info 中的键
        self.now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        
        # 用于存储当前回合的经验
        self.current_episode = []

    def add(self, state, action, reward, next_state, done, info):
        """
        添加经验到缓冲区，并在回合结束时应用 HER。
        """
        # 添加当前经验到缓冲区
        self.buffer.append((state, action, reward, next_state, done, info))
        
        # 添加到当前回合的经验列表
        self.current_episode.append((state, action, reward, next_state, done, info))
        
        # 如果回合结束，应用 HER
        if done:
            self.apply_her()
            # 清空当前回合的经验
            self.current_episode = []

    def apply_her(self):
        """
        使用 HER 的 'future' 策略生成并添加替代经验。
        """
        episode_length = len(self.current_episode)
        if episode_length < 2:
            return  # 不足以生成替代经验
        
        for idx, transition in enumerate(self.current_episode):
            # 对每个经验生成 her_k 个替代经验
            for _ in range(self.her_k):
                # 随机选择一个未来的时间步
                future_idx = random.randint(idx + 1, episode_length - 1)
                future_transition = self.current_episode[future_idx]
                achieved_goal = self.extract_goal(future_transition[3], future_transition[5])  # next_state, info
                
                # 创建新的状态和下一个状态，替换目标
                new_state = self.replace_goal(transition[0], achieved_goal)
                new_next_state = self.replace_goal(transition[3], achieved_goal)
                
                # 重新计算奖励和 done
                new_reward, new_done = self.compute_reward(new_next_state, achieved_goal)
                
                # 创建新的 info
                new_info = transition[5].copy()
                new_info['is_her'] = True  # 标记这是通过 HER 生成的经验
                new_info[self.goal_key] = achieved_goal  # 更新目标
                
                # 添加替代经验到缓冲区
                self.buffer.append((new_state, transition[1], new_reward, new_next_state, new_done, new_info))

    def extract_goal(self, state, info):
        """
        从状态或 info 中提取目标。

        参数：
        - state: 当前状态。
        - info: info 字典。

        返回：
        - goal: 提取的目标。
        """
        if self.goal_key in state:
            return state[self.goal_key]
        elif self.goal_key in info:
            return info[self.goal_key]
        else:
            raise KeyError(f"无法在状态或 info 中找到目标键 '{self.goal_key}'")

    def replace_goal(self, state, new_goal):
        """
        替换状态中的目标为新的目标。

        参数：
        - state: 原始状态。
        - new_goal: 新的目标。

        返回：
        - new_state: 替换后的新状态。
        """
        new_state = state.copy()
        if self.goal_key in new_state:
            new_state[self.goal_key] = new_goal
        elif self.goal_key in new_state.get('info', {}):
            new_state['info'][self.goal_key] = new_goal
        else:
            # 如果目标不在状态中，可以将其添加到 info 中
            new_state['info'] = new_state.get('info', {}).copy()
            new_state['info'][self.goal_key] = new_goal
        return new_state

    def compute_reward(self, state, goal):
        """
        根据新的目标重新计算奖励和 done 标志。

        这是一个占位函数，您需要根据具体的环境实现它。

        参数：
        - state: 当前状态。
        - goal: 新的目标。

        返回：
        - reward: 重新计算的奖励。
        - done: 重新计算的 done 标志。
        """
        # 这里假设环境有一个 compute_reward 方法
        # 您需要根据实际情况实现这一部分
        # 例如：
        # return self.env.compute_reward(state, goal), self.env.is_done(state, goal)
        # 为了示例，我们假设目标状态与当前状态相等时奖励为1，否则为0
        achieved = np.array_equal(state, goal)
        reward = 1.0 if achieved else 0.0
        done = achieved
        return reward, done

    def sample(self, batch_size):
        """
        按照以下规则采样：
        1. 至少 20% 的样本为成功案例（info = {"reaction" : "success"}）。
           如果成功案例不足，则取出所有成功案例。
        2. 剩余的 80% 样本中，40% 为最新经验，40% 为随机采样。
        3. 包含 HER 生成的经验。
        """
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if len(self.buffer) == 0:
            raise ValueError("ReplayBuffer 为空，无法采样")
        
        # 定义采样比例
        success_ratio = 0.2
        latest_ratio = 0.4
        random_ratio = 0.4
        
        num_success = int(success_ratio * batch_size)
        num_latest = int(latest_ratio * batch_size)
        num_random = batch_size - num_success - num_latest  # 确保总数为 batch_size
        
        # 获取所有成功案例的索引
        success_indices = [idx for idx, experience in enumerate(self.buffer) 
                           if experience[-1].get("reaction") == "success" or experience[-1].get("is_her", False)]
        
        # 实际成功案例数量
        actual_success = min(num_success, len(success_indices))
        
        # 采样成功案例
        if actual_success > 0:
            sampled_success = random.sample(success_indices, actual_success)
            sampled_success_experiences = [self.buffer[idx] for idx in sampled_success]
        else:
            sampled_success_experiences = []
            actual_success = 0
        
        # 采样最新经验
        sampled_latest_experiences = []
        if num_latest > 0:
            sampled_latest_experiences = list(self.buffer)[-num_latest:]
        
        # 计算随机采样的数量
        remaining = batch_size - actual_success - len(sampled_latest_experiences)
        if remaining > 0:
            # 获取可供随机采样的索引，排除已经采样的成功案例和最新经验
            exclude_indices = set()
            exclude_indices.update(sampled_success)
            exclude_indices.update(range(len(self.buffer) - num_latest, len(self.buffer)))
            
            available_indices = list(set(range(len(self.buffer))) - exclude_indices)
            
            # 如果可供随机采样的样本不足，允许重复采样
            if len(available_indices) < remaining:
                sampled_random_experiences = random.choices(self.buffer, k=remaining)
            else:
                sampled_random = random.sample(available_indices, remaining)
                sampled_random_experiences = [self.buffer[idx] for idx in sampled_random]
        else:
            sampled_random_experiences = []
        
        # 合并所有采样的经验
        batch = sampled_success_experiences + sampled_latest_experiences + sampled_random_experiences
        
        # 如果总样本数量不足，进行补充
        if len(batch) < batch_size:
            additional = batch_size - len(batch)
            additional_samples = random.choices(self.buffer, k=additional)
            batch += additional_samples
        
        # 打乱批次中的样本
        random.shuffle(batch)
        
        # 解压批次中的数据
        states, actions, rewards, next_states, dones, infos = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def size(self):
        return len(self.buffer)

def change_image_color2gray(image_folder_path):
    """read all image from a folder and change them to gray"""
    # generate a new folder to save those gray images
    new_folder_path = image_folder_path + "_gray"
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    # read all images from the folder
    for image_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_name)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_image_path = os.path.join(new_folder_path, image_name)
        cv2.imwrite(new_image_path, gray_image)
    print("All images have been changed to gray successfully!")


def point_in_rotated_rect_polar(
    result,
    image_shape,
    rect_center,
    rect_angle_deg,
    rect_side_length
):
    """
    将鼠标点击得到的矩阵相对坐标转换到旋转矩形坐标系下的极坐标。
    参数说明：
    1. result: (x_frac, y_frac)，矩阵相对坐标(0~1)。
    2. image_shape: (height, width)。
    3. rect_center: (cx, cy)，旋转矩形中心像素坐标。
    4. rect_angle_deg: 旋转矩形的角度(度数)，0 表示不旋转，正值代表逆时针旋转。
    5. rect_side_length: 旋转矩形边长像素长度。

    返回： (r_frac, theta_deg)
    其中：
    - r_frac：半径相对于 rect_side_length 的归一化值 (0~1)。
    - theta_deg：极坐标角度 (0~360)。
    """
    height, width = image_shape[:2]
    px = result[0] * width
    py = result[1] * height

    dx = px - rect_center[0]
    dy = py - rect_center[1]

    angle_rad = np.deg2rad(-rect_angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a

    r = np.sqrt(dx_rot**2 + dy_rot**2)
    theta_deg = np.degrees(np.arctan2(dy_rot, dx_rot))
    theta_deg = (theta_deg + 360) % 360

    r_frac = r / rect_side_length
    return r_frac, theta_deg, dx_rot, dy_rot

def visualize_polar_coordinate():
    # 生成空白图以便演示
    img_height = 700
    img_width = 700
    image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 220

    # 设置旋转矩形参数
    rect_center = (350, 350)          # 中心像素坐标
    rect_side_length = 400            # 边长
    rect_angle_deg = 45               # 旋转角度
    
    # 在此示例中，假设用户点击位置的相对坐标
    result = (0.3, 0.3)  # (x_frac, y_frac)

    # 把旋转矩形先画出来
    # 1. 矩形的宽高都取 rect_side_length
    width = rect_side_length
    height = rect_side_length
    box = cv2.boxPoints(((rect_center[0], rect_center[1]), (width, height), rect_angle_deg))
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # 计算 result 对应的极坐标
    r_frac, theta_deg, dx_rot, dy_rot = point_in_rotated_rect_polar(
        result,
        (img_height, img_width),
        rect_center,
        rect_angle_deg,
        rect_side_length
    )

    print(f'dx_rot={dx_rot:.2f}, dy_rot={dy_rot:.2f}')
    # 将相对坐标转为像素坐标
    px = int(result[0] * img_width)
    py = int(result[1] * img_height)

    # 在图上标出点击点
    cv2.circle(image, (px, py), 5, (0, 0, 255), -1)

    # 在图上标出中心点
    cv2.circle(image, rect_center, 5, (255, 0, 0), -1)

    # 在图上显示极坐标信息
    text = f"r_frac={r_frac:.2f}, theta_deg={theta_deg:.2f}"
    cv2.putText(
        image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )

    # 显示图像
    cv2.imshow("Visualize Rotated Rect Polar Coord", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalize_state(state):
    legal_fronts = {
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 1, 1, 0),
        (1, 1, 1, 1)
    }
    a, b, c, d, path = state
    current_front = (a, b, c, d)
    
    if current_front in legal_fronts:
        return state.copy(), []  # 无需变换
    
    # 所有可能的变换及其应用函数
    transforms = [
        ('rotate90', lambda a, b, c, d: (d, a, b, c)),
        ('rotate180', lambda a, b, c, d: (c, d, a, b)),
        ('rotate270', lambda a, b, c, d: (b, c, d, a)),
        ('flip_x', lambda a, b, c, d: (d, c, b, a)),
        ('flip_y', lambda a, b, c, d: (b, a, d, c)),
        ('flip_yx', lambda a, b, c, d: (a, d, c, b)),
        ('flip_ynx', lambda a, b, c, d: (a, c, b, d)),
    ]
    
    # 尝试所有变换
    for name, trans_fn in transforms:
        transformed = trans_fn(a, b, c, d)
        if transformed in legal_fronts:
            normalized = list(transformed) + [path]
            return np.array(normalized), [name]
    
    # 组合变换（示例：先旋转再翻转）
    for t1 in transforms:
        for t2 in transforms:
            # 应用t1然后t2
            t1_name, t1_fn = t1
            t2_name, t2_fn = t2
            temp = t1_fn(a, b, c, d)
            transformed = t2_fn(*temp)
            if transformed in legal_fronts:
                normalized = list(transformed) + [path]
                return np.array(normalized), [t1_name, t2_name]
    
    raise ValueError(f"Cannot normalize state: {state}")

if __name__ == '__main__':

    # hyperparameters = {
    #     "buffer_size": 1000,
    #     "her_k": 4,
    #     "goal_key": "reaction"
    # }
    # str_hyperparameters = '_'.join([f"{key}={value}" for key, value in hyperparameters.items()])
    # print(f"Hyperparameters: {str_hyperparameters}")

    # change_image_color2gray('./058a_images/1118')
    
    nanonis = NanonisController()
    # nanonis.ScanStart()
    # nanonis.Home()
    # while True:
    #     nanonis.ScanFrameData(14)
    #     print("get data")

    # visualize_polar_coordinate()

    
    # state = np.array([0, 0, 1, 1, True])
    # normalized_state, transforms = normalize_state(state)
    # print(f"Normalized: {normalized_state}, Transforms: {transforms}")


    # img = io.imread('./test_seg_img/Scan_data_back2025-03-20 23-47-58.png', as_gray=True)

    # # 对比度拉伸
    # p2, p98 = np.percentile(img, (2, 98))
    # stretched = exposure.rescale_intensity(img, in_range=(p2, p98))

    # # 伽马校正
    # gamma_corrected = exposure.adjust_gamma(img, gamma=2)

    # # 将原图、对比度拉伸后的图像和伽马校正后的图像拼接在一起以便可视化对比
    # combined = np.hstack([img, stretched, gamma_corrected])

    # # 使用 matplotlib 进行可视化
    # plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 1)
    # plt.title("Original")
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.title("Contrast Stretched")
    # plt.imshow(stretched, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.title("Gamma Corrected")
    # plt.imshow(gamma_corrected, cmap='gray')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()
    

    
    # print(np.array([1, 2, 3, 4, 5])[:4])

    # nanonis = NanonisController()

    # tip_induce_mode = 'CC'  # 'CC' or 'CH'
    # tip_bias = 3.45  # V
    # tip_current = 0.7  # nA
    # lift_time = 4  # s
    # max_hold_time = 20  # s
    # stop_current_threshold = 0.3  # nA

    # tip_current = tip_current*1e-9  # convert to A
    # print('Start the tip manipulation...')
    # # print(f'Tip position:  X:{action_from_SAC[0]}, Y:{action_from_SAC[1]}')
    # # print(f'Tip Bias: {action_from_SAC[2]}V, Tip Current: {action_from_SAC[3]}nA')
    # # do the tip manipulation
    # tip_bias_init = nanonis.BiasGet()
    # tip_current_init = nanonis.SetpointGet()

    # abs_tip_bias_init = abs(tip_bias_init)
    # abs_tip_current_init = abs(tip_current_init)

    # nanonis.BiasSet(abs_tip_bias_init)
    # nanonis.SetpointSet(abs_tip_current_init)

    # if tip_induce_mode == 'CC':
    #     # set the tip bias and current should be changed in 4s
    #     steps = 50  # 分成50步
    #     time_interval = lift_time / steps  # 每步的时间间隔

    #     # 计算每步的增量
    #     bias_step = (tip_bias - abs_tip_bias_init) / steps
    #     current_step = (tip_current - abs_tip_current_init) / steps

    #     # nanonis.ZCtrlOff()

    #     for i in range(steps):
    #         # 逐步设置Bias和Setpoint
    #         nanonis.BiasSet(abs_tip_bias_init + bias_step * (i + 1))
    #         nanonis.SetpointSet(abs_tip_current_init + current_step * (i + 1))
    #         time.sleep(time_interval)
        
    #     nanonis.ZCtrlOff()

    #     # time.sleep(12)   # wait for the tip induce

    #     start_time = time.time()
    #     signal_history = deque(maxlen=10)
    #     while True:
    #         # 保留两位小数
    #         signal_current = nanonis.SignalValsGet(0)['0']*1e9
    #         signal_current = round(signal_current, 4)
    #         signal_history.append(signal_current)  # 将当前值添加到deque中
    #         # print(signal_current)   

    #         # 检查是否已经超过 15 秒
    #         if time.time() - start_time > max_hold_time :
    #             print(f"The loop has been running for {max_hold_time} seconds, exiting the tip induce.")
    #             break
    #         # if signal_current < stop_current_threshold:
    #         #     print(f"The loop has been running for {time.time() - start_time} seconds, exiting the tip induce.")
    #         #     break
    #         # 如果deque中的所有值都小于stop_current_threshold，则跳出循环
    #         if len(signal_history) == signal_history.maxlen and all(value < stop_current_threshold for value in signal_history):
    #             print(f"The loop has been running for {time.time() - start_time} seconds, exiting the tip induce.")
    #             break    

    #     # initialize the tip bias and current
    #     # nanonis.ZCtrlOff()
    #     nanonis.BiasSet(tip_bias_init)
    #     nanonis.SetpointSet(tip_current_init)
    #     nanonis.ZCtrlOnSet()
    
    print('Start the tip manipulation...')
    lift_time = 0.5  # s
    max_hold_time = 20  # s
    pulse_time = 0.1  # s
    tip_induce_mode = 'pulse'  # 'CC' or 'CH'

    v = '-4.1'
    i = '0.07n'
    tip_bias = nanonis.convert(v)  # convert to V
    tip_current = nanonis.convert(i)  # convert to nA

    tip_bias_init = nanonis.BiasGet()
    tip_current_init = nanonis.SetpointGet()

    abs_tip_bias_init = abs(tip_bias_init)
    abs_tip_current_init = abs(tip_current_init)

    
    if tip_induce_mode == 'pulse':

        # set the tip bias and current should be changed in 4s
        steps = 20  # 分成50步
        time_interval = lift_time / steps  # 每步的时间间隔

        # 计算每步的增量
        current_step = (tip_current - abs_tip_current_init) / steps
        

        # nanonis.ZCtrlOff()

        for i in range(steps):
            # 逐步设置Setpoint
            nanonis.SetpointSet(abs_tip_current_init + current_step * (i + 1))
            time.sleep(time_interval)
        
        # nanonis.ZCtrlOff()

        # 初始化参数
        pre_pulse_duration = 0.5  # 前0.5秒提取信号
        post_pulse_duration = 0.5  # 后0.5秒提取信号
        max_iterations = 5  # 最大循环次数
        threshold = '20p'  # 插值阈值
        threshold = nanonis.convert(threshold)*1e9  # 将阈值转换为纳米单位
        pulse_time = 0.1  # nanonis.pulse的持续时间

        # 开始循环
        for iteration in range(max_iterations):
            # 提取前0.5秒的信号
            pre_pulse_signals = []
            start_time = time.time()
            while time.time() - start_time < pre_pulse_duration:
                signal_current = nanonis.SignalValsGet(30)['0'] * 1e9  # 获取信号值
                pre_pulse_signals.append(signal_current)
                time.sleep(0.01)  # 防止过于频繁的采样

            # 计算前0.5秒信号的平均值
            pre_pulse_avg = sum(pre_pulse_signals) / len(pre_pulse_signals)
            time.sleep(0.1)  # 等待0.5秒
            # 施加脉冲
            nanonis.BiasPulse(tip_bias, pulse_time)
            time.sleep(0.1)  # 等待脉冲施加完成
            # 提取后0.5秒的信号
            post_pulse_signals = []
            start_time = time.time()
            while time.time() - start_time < post_pulse_duration:
                signal_current = nanonis.SignalValsGet(30)['0'] * 1e9  # 获取信号值
                post_pulse_signals.append(signal_current)
                time.sleep(0.01)  # 防止过于频繁的采样

            # 计算后0.5秒信号的平均值
            post_pulse_avg = sum(post_pulse_signals) / len(post_pulse_signals)

            # 计算插值
            signal_difference = abs(post_pulse_avg - pre_pulse_avg)

            print(f"Iteration {iteration + 1}: Pre-pulse Avg = {pre_pulse_avg}, Post-pulse Avg = {post_pulse_avg}, Difference = {signal_difference}")

            # 判断是否满足中止条件
            if signal_difference > threshold:
                print("Signal difference exceeds threshold. Exiting loop.")
                break

        # 如果达到最大循环次数仍未满足条件，强制跳出
        if iteration == max_iterations - 1:
            print("Maximum iterations reached. Forcing exit.")
        # nanonis.BiasSet(tip_bias_init)
        nanonis.SetpointSet(tip_current_init)
        # nanonis.ZCtrlOnSet()

    time.sleep(1)
    print('Tip manipulation is done.')
    pass

    # from keypoint.detect import key_detect
    # img_simu_7_path = './STM_img_simu/TPM_image/001.png'
    # keypoint_model_path = './keypoint/best_Ni.pt'
    # key_point_save_dir = './keypoint/TPM_result/'
    # image_for = cv2.imread(img_simu_7_path, cv2.IMREAD_GRAYSCALE)           # read the simu image
    # image_for = cv2.resize(image_for, (304, 304), interpolation=cv2.INTER_AREA)
    # key_points_result = key_detect(image_for, keypoint_model_path, key_point_save_dir)
    # print(key_points_result)
    # save the key_points_result as txt