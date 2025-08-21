import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# 如果需要使用 skimage 的 remove_small_objects / remove_small_holes，请先:
# pip install scikit-image
from skimage.morphology import remove_small_holes, remove_small_objects

from EvaluationCNN.detect import predict_image_quality
from keypoint.detect import key_detect
from mol_segment.detect import segmented_image

matplotlib.use('Agg')

def generate_square_vertices(center, side_length, angle):
    """
    生成正方形的四个顶点坐标。

    参数:
    center (tuple): 正方形中心点坐标 (x, y)。
    side_length (float): 正方形的边长。
    angle (float): 正方形的旋转角度（以度为单位）。

    返回:
    numpy.ndarray: 正方形的四个顶点坐标。
    """
    half_side = side_length / 2

    # 定义未旋转的正方形顶点
    vertices = np.array([
        [-half_side, -half_side],
        [half_side, -half_side],
        [half_side, half_side],
        [-half_side, half_side]
    ])

    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)

    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # 旋转顶点
    rotated_vertices = np.dot(vertices, rotation_matrix)

    # 平移顶点到中心点
    rotated_vertices += center

    return rotated_vertices

def calculate_orientation(mask):
    # 查找mask的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 获取最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    
    # 获取矩形的顶点
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # 计算角度
    angle = rect[-1]
    
    # 调整角度范围
    if angle < -45:
        angle += 90
    
    return angle

def estimate_mol_orientation(
        mask,
        min_obj_size=100,
        hole_area_threshold=500,
        close_kernel_size=20,
        img_scale= "10n",
        mol_scale = "2n",
        key_point_result = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        square_save_path = "./"
    ):
    """
    使用 findContours + minAreaRect 获取目标主方向的版本。
    
    1) 去除小块噪声 (remove_small_objects)
    2) 填补小孔洞 (remove_small_holes)
    3) 形态学闭运算 (cv2.MORPH_CLOSE)
    4) 连通分量分析, 只保留最大块 (可选, 也可直接对预处理后的结果做 findContours)
    5) findContours + minAreaRect 获取目标主方向
    6) Matplotlib 可视化

    参数:
    mask               : 二值掩膜(H, W)，值域可 0/1 或 0/255
    min_obj_size       : 去除面积小于该值的小块
    hole_area_threshold: 填补面积小于该值的孔洞
    close_kernel_size  : 闭运算kernel大小，用于填补大缝隙；<=1则跳过

    返回:
    angle_degrees : 分子的主要方向角度(度)，范围 [0, 90)。若无法计算则返回 None。
                    (minAreaRect 返回的角度本身区间一般在 [-90, 0)，此处做了修正。)
    """

    # ========== 0) 将 mask 转成 0/1 ==========
    if mask.max() > 1:
        bin_mask = (mask > 0).astype(np.uint8)
    else:
        bin_mask = mask.astype(np.uint8)
    raw_mask = bin_mask.copy()  # 备份原始用于可视化

    # ========== 1) 去除小块噪声、填孔、闭运算 ==========
    # 1.1 转布尔用于 skimage
    bool_mask = (bin_mask > 0)
    # a) 去除面积太小的块
    bool_removed_small = remove_small_objects(bool_mask, min_size=min_obj_size)
    # b) 填补小孔洞
    bool_filled_holes = remove_small_holes(bool_removed_small, area_threshold=hole_area_threshold)
    pre_mask = bool_filled_holes.astype(np.uint8)

    # c) 闭运算(可选)
    if close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        pre_mask_closed = cv2.morphologyEx(pre_mask, cv2.MORPH_CLOSE, kernel)
    else:
        pre_mask_closed = pre_mask.copy()

    # ========== 2) 连通分量分析, 保留最大块 (可选, 让轮廓更干净) ==========
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pre_mask_closed, connectivity=8)
    if num_labels <= 1:
        # 没有前景
        largest_mask = np.zeros_like(pre_mask_closed)
    else:
        max_area = 0
        max_label = 0
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = label_id
        largest_mask = (labels == max_label).astype(np.uint8)

    # ========== 3) 利用 findContours + minAreaRect 获取主方向 ==========
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        angle_degrees = None
    else:
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        # 最小外接矩形
        rect = cv2.minAreaRect(largest_contour)  # ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # rect[-1] 给出角度
        angle = rect[-1]
        # angle = 90
        # minAreaRect 返回的 angle 通常在 [-90, 0)；若旋转矩形竖着，会出现 -45 之类
        # 这里做个修正，使 angle 在 [0, 90) (具体需求可根据项目自行调整)

        if angle < 0:
            angle += 90
        if angle >= 90:
            angle -= 90
        if angle == 0:
            angle = 0.01            
        angle_degrees = angle

    # ========== 4) Matplotlib 可视化过程 (分 2x3 六个子图) ==========
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    # (0) 原始 mask
    axes[0].imshow(raw_mask, cmap='gray')
    axes[0].set_title('0) Raw Mask')
    axes[0].axis('off')

    # (1) 去除小块噪声
    axes[1].imshow(bool_removed_small, cmap='gray')
    axes[1].set_title('1) Remove Small Objects')
    axes[1].axis('off')

    # (2) 填补小孔洞
    axes[2].imshow(bool_filled_holes, cmap='gray')
    axes[2].set_title('2) Fill Small Holes')
    axes[2].axis('off')

    # (3) 闭运算
    axes[3].imshow(pre_mask_closed, cmap='gray')
    axes[3].set_title('3) Closing')
    axes[3].axis('off')

    # (4) 最大连通分量
    axes[4].imshow(largest_mask, cmap='gray')
    axes[4].set_title('4) Largest Component')
    axes[4].axis('off')

    # (5) 最终显示最小外接矩形
    vis_img = cv2.cvtColor(largest_mask * 255, cv2.COLOR_GRAY2BGR)
    if angle_degrees is not None:
        # 画出最大轮廓
        cv2.drawContours(vis_img, [largest_contour], -1, (0, 255, 0), 2)
        # 画出外接矩形
        cv2.drawContours(vis_img, [box], -1, (0, 0, 255), 2)
        # 画出最终拟合框
        mask_edge_pix = mask.shape[0]  # mask 的边长
        mol_box_length = mask_edge_pix/(img_scale/mol_scale) # 分子盒子的像素长度
        mol_box = generate_square_vertices(tuple(np.array(key_point_result[0][5:7])*mask_edge_pix), mol_box_length, -angle_degrees)
        mol_box = mol_box.astype(np.int32)
        cv2.drawContours(vis_img, [mol_box], -1, (255, 0, 0), 2)

        axes[5].set_title(f'5) Orientation={angle_degrees:.2f}°')
    else:
        axes[5].set_title('5) No valid contour')
    axes[5].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    axes[5].axis('off')
    # 画一个箭头指示角度
    if angle_degrees is not None:
        (cx, cy), _, _ = rect
        dx = np.cos(np.radians(angle_degrees))
        dy = np.sin(np.radians(angle_degrees))
        axes[5].arrow(cx, cy, dx*20, dy*20, color='r', width=2)

    plt.tight_layout()
    # 获取当前时间戳
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # 判断保存路径是否存在
    if not os.path.exists(square_save_path):
        os.makedirs(square_save_path)
    
    plt.savefig(square_save_path + f"/{time_stamp}.png")
    plt.clf()

    return angle_degrees

def extract_reaction_site_patches_vis(  
        image, 
        key_point_result = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        img_scale = 1000 ,
        mol_scale = 200,
        Br_size = 60, 
        orientation_angle = 0, 
        save_dir="./reaction_patches",
        save_patch = True
        ):
    image_pix_edge = image.shape[0]
    Br_site_edge = image_pix_edge/(img_scale/Br_size)
    mol_edge = image_pix_edge/(img_scale/mol_scale)
    # 1. 计算分子四个反应位点的位置
    molecule_center = np.array(key_point_result[0][5:7]) * image_pix_edge
    # 假设分子为一个正方形，分子大小已知，可以计算反应位点坐标
    half_size = mol_edge / 2
    # 生成四个顶点的相对坐标（基于分子中心）
    reaction_points = [
        (molecule_center[0] - half_size, molecule_center[1] - half_size),
        (molecule_center[0] + half_size, molecule_center[1] - half_size),
        (molecule_center[0] + half_size, molecule_center[1] + half_size),
        (molecule_center[0] - half_size, molecule_center[1] + half_size)
    ]
    
    # 2. 对反应位点进行旋转，确保对齐分子的实际取向
    orientation_angle = -1*orientation_angle
    rotation_matrix = cv2.getRotationMatrix2D(molecule_center, orientation_angle, 1)
    rotated_reaction_points = []
    for pt in reaction_points:
        rotated_pt = cv2.transform(np.array([[pt]], dtype=np.float32), rotation_matrix)[0][0]
        rotated_reaction_points.append(rotated_pt)
    
    # 可视化步骤
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    
    # 绘制分子中心
    ax.scatter(molecule_center[0], molecule_center[1], color='red', label="Center", zorder=5)
    
    # 绘制反应位点
    for pt in rotated_reaction_points:
        ax.scatter(pt[0], pt[1], color='blue', label="Br_site", zorder=5)
    
    # 画出分子的位置和旋转
    molecule_square = plt.Polygon(rotated_reaction_points, fill=None, edgecolor='yellow', linewidth=2)
    ax.add_patch(molecule_square)

    ax.set_title("Molecule and Reaction Points Visualization")
    ax.legend()

    # 3. 生成patch，边长为40个像素
    patch_size = round(Br_site_edge)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    patches = []
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    for i, pt in enumerate(rotated_reaction_points):
        # 计算每个反应位点附近的正方形区域
        x, y = int(pt[0]), int(pt[1])
        x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
        x2, y2 = min(image.shape[1], x + patch_size // 2), min(image.shape[0], y + patch_size // 2)
        
        # 裁切出patch
        patch = image[y1:y2, x1:x2]
        
        # 保存patch
        if save_patch:
            patch_filename = os.path.join(save_dir, f"reaction_point_{i+1}_{time_stamp}.png")
            cv2.imwrite(patch_filename, patch)
        patches.append(patch)
    
    # 绘制反应位点的patch
    for i, pt in enumerate(rotated_reaction_points):
        x, y = int(pt[0]), int(pt[1])
        x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
        x2, y2 = min(image.shape[1], x + patch_size // 2), min(image.shape[0], y + patch_size // 2)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    # plt.savefig(save_dir + f"/reaction_point_{time_stamp}.png")
    # clear the plot
    plt.clf()
    return patches

def Br_site_patches_vis(  
        image, 
        key_point_result = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        img_scale = 1000 ,
        mol_scale = 200,
        Br_size = 60, 
        orientation_angle = 0, 
        save_dir="./reaction_patches",
        save_patch = False
        ):
    image_pix_edge = image.shape[0]
    Br_site_edge = image_pix_edge/(img_scale/Br_size)
    mol_edge = image_pix_edge/(img_scale/mol_scale)
    # 1. 计算分子四个反应位点的位置
    molecule_center = np.array(key_point_result[0][5:7]) * image_pix_edge
    # 假设分子为一个正方形，分子大小已知，可以计算反应位点坐标
    half_size = mol_edge / 2
    # 生成四个顶点的相对坐标（基于分子中心）
    reaction_points = [
        (molecule_center[0] - half_size, molecule_center[1] - half_size), #
        (molecule_center[0] + half_size, molecule_center[1] - half_size),
        (molecule_center[0] + half_size, molecule_center[1] + half_size),
        (molecule_center[0] - half_size, molecule_center[1] + half_size)
    ]
    
    # 2. 对反应位点进行旋转，确保对齐分子的实际取向
    orientation_angle = -1*orientation_angle
    rotation_matrix = cv2.getRotationMatrix2D(molecule_center, orientation_angle, 1)
    rotated_reaction_points = []
    for pt in reaction_points:
        rotated_pt = cv2.transform(np.array([[pt]], dtype=np.float32), rotation_matrix)[0][0]
        rotated_reaction_points.append(rotated_pt)
    
    # 可视化步骤
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    
    # 绘制分子中心
    ax.scatter(molecule_center[0], molecule_center[1], color='red', label="Center", zorder=5)
    
    # 绘制反应位点
    for pt in rotated_reaction_points:
        ax.scatter(pt[0], pt[1], color='blue', label="Br_site", zorder=5)
    
    # 画出分子的位置和旋转
    molecule_square = plt.Polygon(rotated_reaction_points, fill=None, edgecolor='yellow', linewidth=2)
    ax.add_patch(molecule_square)

    ax.set_title("Molecule and Reaction Points Visualization")
    ax.legend()

    # 3. 生成patch，边长为40个像素
    patch_size = round(Br_site_edge)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    patches = []
    Br_site_state_list = []
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    for i, pt in enumerate(rotated_reaction_points):
        # 计算每个反应位点附近的正方形区域
        x, y = int(pt[0]), int(pt[1])
        x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
        x2, y2 = min(image.shape[1], x + patch_size // 2), min(image.shape[0], y + patch_size // 2)
        
        # 裁切出patch
        patch = image[y1:y2, x1:x2]
        predict = predict_image_quality(patch, site_model_path)
        Br_site_state_list.append(predict)
        if predict < 0.3:
            box_color = 'green'
        else:
            box_color = 'red'
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, edgecolor=box_color, linewidth=1)
        # attach the number of the Br site
        ax.text(x1, y1, f"{i+1}", fontsize=12, color=box_color)
        ax.add_patch(rect)
    
        # 保存patch?
        if save_patch:
            patch_filename = os.path.join(save_dir, f"reaction_point_{i+1}_{time_stamp}.png")
            cv2.imwrite(patch_filename, patch)
        patches.append(patch)
    plt.savefig(save_dir + f"/reaction_point_{time_stamp}.png")
    plt.clf()
    return patches


def process_reaction_points(image, key_point_result, img_scale=1000, mol_scale=200, Br_size=60, orientation_angle=0, site_model_path = "./model/site_model.pth"):
    image_pix_edge = image.shape[0]
    Br_site_edge = image_pix_edge / (img_scale / Br_size)
    mol_edge = image_pix_edge / (img_scale / mol_scale)
    
    # 1. Calculate the molecule's reaction points
    molecule_center = np.array(key_point_result[0][5:7]) * image_pix_edge
    half_size = mol_edge / 2
    reaction_points = [
        (molecule_center[0] - half_size, molecule_center[1] - half_size),
        (molecule_center[0] + half_size, molecule_center[1] - half_size),
        (molecule_center[0] + half_size, molecule_center[1] + half_size),
        (molecule_center[0] - half_size, molecule_center[1] + half_size)
    ]
    
    # 2. Rotate the reaction points according to the molecule's orientation
    orientation_angle = -1 * orientation_angle
    rotation_matrix = cv2.getRotationMatrix2D(molecule_center, orientation_angle, 1)
    rotated_reaction_points = []
    for pt in reaction_points:
        rotated_pt = cv2.transform(np.array([[pt]], dtype=np.float32), rotation_matrix)[0][0]
        rotated_reaction_points.append(rotated_pt)
    
    # 3. Prepare patches
    patch_size = round(Br_site_edge)
    patches = []
    Br_site_state_list = []
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    for i, pt in enumerate(rotated_reaction_points):
        # Calculate the square area around each reaction point
        x, y = int(pt[0]), int(pt[1])
        x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
        x2, y2 = min(image.shape[1], x + patch_size // 2), min(image.shape[0], y + patch_size // 2)
        
        # Extract patch
        patch = image[y1:y2, x1:x2]
        # Predict the reaction site state
        predict = predict_image_quality(patch, site_model_path)
        Br_site_state_list.append(predict)
        
        patches.append((patch, (x1, y1, x2, y2), predict))
    
    return rotated_reaction_points, patches, Br_site_state_list, time_stamp


def visualize_reaction_points(image, rotated_reaction_points, patches, save_dir="./reaction_patches", save_patch=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    
    # Draw molecule center
    molecule_center = np.array(rotated_reaction_points[0])  # Assuming molecule center is the first reaction point
    ax.scatter(molecule_center[0], molecule_center[1], color='red', label="Center", zorder=5)
    
    # Draw reaction points
    for pt in rotated_reaction_points:
        ax.scatter(pt[0], pt[1], color='blue', label="Br_site", zorder=5)
    
    # Draw the molecule's bounding box
    molecule_square = plt.Polygon(rotated_reaction_points, fill=None, edgecolor='yellow', linewidth=2)
    ax.add_patch(molecule_square)

    ax.set_title("Molecule and Reaction Points Visualization")
    ax.legend()

    # Draw rectangles around the reaction points and save patches
    for i, (patch, (x1, y1, x2, y2), predict) in enumerate(patches):
        box_color = 'green' if predict < 0.3 else 'red'
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, edgecolor=box_color, linewidth=1)
        ax.text(x1, y1, f"{i+1}", fontsize=12, color=box_color)
        ax.add_patch(rect)
        
        if save_patch:
            patch_filename = os.path.join(save_dir, f"reaction_point_{i+1}_{time_stamp}.png")
            cv2.imwrite(patch_filename, patch)

    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    plt.savefig(save_dir + f"/reaction_point_{time_stamp}.png")
    plt.clf()







def mol_Br_site_detection(img, mask, key_point_result, img_scale, mol_scale, Br_scale, square_save_path):
    angle = estimate_mol_orientation(
        mask,
        min_obj_size=200,
        hole_area_threshold=200,
        close_kernel_size=10,
        img_scale= img_scale,
        mol_scale = mol_scale,
        key_point_result = key_point_result,
        square_save_path = square_save_path + './orientation')
    # print(f"Estimated orientation angle = {angle}")

    patches = Br_site_patches_vis(
        img, 
        key_point_result = key_point_result,
        img_scale = img_scale, 
        mol_scale = mol_scale,
        Br_size = Br_scale, 
        orientation_angle = angle, 
        save_dir= square_save_path +"./reaction_patches"
        )
    
    return patches


# =======================
# 使用示例
if __name__ == "__main__":



    # image_path = './mol_segment/058_test/state_0/367_5.png'
    input_folder = './mol_segment/058_test/test'
    segment_output_folder = './mol_segment/results1'
    segment_model_path = './mol_segment/unet_model-zzw-Ni_V2.pth'
    keypoint_output_folder = './keypoint/results1'
    keypoint_model_path = './keypoint/best.pt'  
    site_model_path = './EvaluationCNN/CNN_Br_V2.pth'
    square_save_path = './single_mol_results'
    
    img_scale = 1000  # 1000 → 10nm
    mol_scale = 160  # 180 → 1.8nm
    Br_scale = 180  # 180 → 1.8nm

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)

            img_key = cv2.imread(image_path)
            # img = Image.open(image_path).convert('RGB')
            img = img_key
            mask,_,_ = segmented_image(img, segment_output_folder,segment_model_path)
            key_point_result = key_detect(img_key, keypoint_model_path, keypoint_output_folder)
            # print(key_point_result)

            #convert the img to 1 channel
            if len(mask.shape) == 3:
                mask = mask.squeeze().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resize the mask to (208,208)
            mask = cv2.resize(mask, (208, 208), interpolation=cv2.INTER_NEAREST)
            
            patches = mol_Br_site_detection(img, mask, key_point_result, img_scale, mol_scale, Br_scale, square_save_path)
            Br_site_state_list = []
            for i, patch in enumerate(patches):
                site_predict = mol_Br_site_detection(patch, site_model_path)
                Br_site_state_list.append(site_predict)
            print(Br_site_state_list)



