import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.morphology import remove_small_holes, remove_small_objects

matplotlib.use('TkAgg')

def fused_orientation_and_reaction_visualization(
        image, 
        mask, 
        key_point_result=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        image_scale=1000,
        molecule_size=200,
        Br_size=60,
        min_obj_size=100,
        hole_area_threshold=500,
        close_kernel_size=20,
        save_dir="fusion_results"
    ):
    """
    对分子mask和关键点信息进行处理，用 findContours+minAreaRect 计算分子主要方向，同时根据关键点
    计算分子的反应位点，并将所有可视化内容汇总到一张图上。

    参数:
        image: 原始图像（建议为单通道灰度图）
        mask: 分割得到的二值mask，可以为0/1或0/255
        key_point_result: 检测出的关键点信息，列表中的第一个元素[5:7]作为分子中心（归一化坐标）
        image_scale: 图像对应的标定尺度
        molecule_size: 分子在标定下的尺寸（像素尺度）
        Br_size: Br原子对应的尺寸标定（用于patch提取，此处计算patch边长）
        min_obj_size: 去除噪点时最小区域面积阈值
        hole_area_threshold: 填补小孔洞的面积阈值
        close_kernel_size: 形态学闭运算的kernel大小，<=1则跳过
        save_dir: 保存结果图的文件夹路径

    返回:
        angle_degrees: 估计的分子主要方向角 (度)
        reaction_points: 反应位点（旋转后的）的坐标列表
    """

    # ========== 预处理mask ==========
    # 若mask值域大于1，则转换为0/1
    if mask.max() > 1:
        bin_mask = (mask > 0).astype(np.uint8)
    else:
        bin_mask = mask.astype(np.uint8)
    raw_mask = bin_mask.copy()  # 用于可视化

    # 1. 去除小块噪声 & 填补小孔洞
    bool_mask = (bin_mask > 0)
    bool_removed_small = remove_small_objects(bool_mask, min_size=min_obj_size)
    bool_filled_holes = remove_small_holes(bool_removed_small, area_threshold=hole_area_threshold)
    pre_mask = bool_filled_holes.astype(np.uint8)

    # 2. 形态学闭运算（可选）
    if close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        pre_mask_closed = cv2.morphologyEx(pre_mask, cv2.MORPH_CLOSE, kernel)
    else:
        pre_mask_closed = pre_mask.copy()

    # 3. 连通分量分析，保留最大区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pre_mask_closed, connectivity=8)
    if num_labels <= 1:
        largest_mask = np.zeros_like(pre_mask_closed)
        contours = []
    else:
        max_area = 0
        max_label = 0
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = label_id
        largest_mask = (labels == max_label).astype(np.uint8)
        contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ========== 计算主要方向 ==========
    if len(contours) == 0:
        angle_degrees = None
        rect = None
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)  # ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        if angle >= 90:
            angle -= 90
        angle_degrees = angle

    # ========== 计算反应位点 ==========
    image_pix_edge = image.shape[0]
    Br_site_edge = image_pix_edge / (image_scale / Br_size)
    # 分子中心（以图像像素为单位），取关键点中坐标位置 [5:7]
    molecule_center = np.array(key_point_result[0][5:7]) * image_pix_edge
    # 根据 molecule_size 计算对应分子在图像上的边长
    mol_edge = image_pix_edge / (image_scale / molecule_size)
    half_size = mol_edge / 2
    # 未旋转的反应位点（正方形四个顶点）
    reaction_points = [
        (molecule_center[0] - half_size, molecule_center[1] - half_size),
        (molecule_center[0] + half_size, molecule_center[1] - half_size),
        (molecule_center[0] + half_size, molecule_center[1] + half_size),
        (molecule_center[0] - half_size, molecule_center[1] + half_size)
    ]
    # 对反应位点进行旋转对齐分子的实际方向（方向取相反值，与cv2.getRotationMatrix2D的旋转规则一致）
    rot_angle = -angle_degrees if angle_degrees is not None else 0
    rotation_matrix = cv2.getRotationMatrix2D(tuple(molecule_center), rot_angle, 1)
    rotated_reaction_points = []
    for pt in reaction_points:
        rotated_pt = cv2.transform(np.array([[pt]], dtype=np.float32), rotation_matrix)[0][0]
        rotated_reaction_points.append(rotated_pt)

    # ========== 可视化：所有内容汇总到一张图中 ==========
    fig, ax = plt.subplots(figsize=(10, 10))
    # 显示原始图像（若图像为彩色则转换为灰度显示）
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()
    ax.imshow(gray_img, cmap='gray')
    
    # 绘制原始mask边界（以蓝色显示）
    contours_raw, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_raw:
        cnt = cnt.squeeze()
        if cnt.ndim == 2:
            ax.plot(cnt[:, 0], cnt[:, 1], color='blue', linewidth=1, label="Raw Mask")
    
    # 绘制最大连通区域轮廓（绿色）
    if len(contours) > 0:
        cnt = largest_contour.squeeze()
        if cnt.ndim == 2:
            ax.plot(cnt[:, 0], cnt[:, 1], color='green', linewidth=2, label="Largest Component")
    
    # 绘制最小外接矩形（红色）
    if rect is not None:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = np.vstack([box, box[0]])  # 用于闭合多边形
        ax.plot(box[:, 0], box[:, 1], color='red', linewidth=2, label="MinAreaRect")
        # 在矩形中心画一个箭头表示方向
        (cx, cy), _, _ = rect
        dx = np.cos(np.radians(angle_degrees)) * 20 if angle_degrees is not None else 0
        dy = np.sin(np.radians(angle_degrees)) * 20 if angle_degrees is not None else 0
        ax.arrow(cx, cy, dx, dy, color='yellow', width=2)
    
    # 绘制分子中心（红点）
    ax.scatter(molecule_center[0], molecule_center[1], color='red', s=50, label="Molecule Center")
    
    # 绘制旋转后的反应点（蓝点）
    rp_array = np.array(rotated_reaction_points)
    ax.scatter(rp_array[:, 0], rp_array[:, 1], color='blue', s=50, label="Reaction Points")
    # 绘制反应位点构成的多边形（黄色边框）
    poly = plt.Polygon(rotated_reaction_points, fill=None, edgecolor='yellow', linewidth=2, label="Reaction Square")
    ax.add_patch(poly)
    
    # 对每个反应位点周围绘制patch框（patch大小基于Br_site_edge）
    patch_size = round(Br_site_edge)
    for pt in rotated_reaction_points:
        x, y = int(pt[0]), int(pt[1])
        x1 = max(0, x - patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        rect_patch = plt.Rectangle((x1, y1), patch_size, patch_size, fill=None, edgecolor='magenta', linewidth=2)
        ax.add_patch(rect_patch)
    
    # 标题添加估计的角度信息
    ax.set_title(f"Fused Visualization\nEstimated Orientation Angle = {angle_degrees:.2f}°" if angle_degrees is not None else "Fused Visualization\nNo Valid Contour")
    ax.axis('off')
    plt.tight_layout()

    # 保存结果图
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path = os.path.join(save_dir, f"fused_{time_stamp}.png")
    plt.savefig(save_path)
    plt.show()

    return angle_degrees, rotated_reaction_points

# =======================
# 使用示例
if __name__ == "__main__":
    # 测试示例路径（根据实际情况修改）
    image_path = './mol_segment/058_test/258_10.png'
    segment_output_folder = './mol_segment/results1'
    segment_model_path = './mol_segment/unet_model-zzw-Ni_V2.pth'
    keypoint_output_folder = './keypoint/results1'
    keypoint_model_path = './keypoint/best.pt'
    square_save_path = './single_mol_results'

    # 读取图像（使用 cv2.imread 得到BGR图像）
    img_key = cv2.imread(image_path)
    img = img_key
    # 调用分割函数获取mask（请确保 segmented_image 函数已定义）
    from mol_segment.detect import segmented_image
    mask, _, _ = segmented_image(img, segment_output_folder, segment_model_path)
    # 检测关键点（请确保 key_detect 函数已定义）
    from keypoint.detect import key_detect
    key_point_result = key_detect(img_key, keypoint_model_path, keypoint_output_folder)
    print("Key Point Result:", key_point_result)

    if len(mask.shape) == 3:
        mask = mask.squeeze().astype(np.uint8)
    # resize mask为固定尺寸（例如 208 x 208）
    mask = cv2.resize(mask, (208, 208), interpolation=cv2.INTER_NEAREST)

    # 转为灰度图（如果img是BGR）
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 调用融合函数
    angle, reaction_pts = fused_orientation_and_reaction_visualization(
                                gray_img,
                                mask,
                                key_point_result=key_point_result,
                                image_scale=1000,
                                molecule_size=180,
                                Br_size=180,
                                min_obj_size=200,
                                hole_area_threshold=200,
                                close_kernel_size=10,
                                save_dir="fusion_results"
                            )
    print(f"Estimated Orientation Angle = {angle}")