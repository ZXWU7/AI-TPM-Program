import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# 如果需要使用 skimage 的 remove_small_objects / remove_small_holes，请先:
# pip install scikit-image
from skimage.morphology import remove_small_holes, remove_small_objects

from keypoint.detect import key_detect
from mol_segment.detect import segmented_image


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

def estimate_and_visualize(
    mask,
    min_obj_size=50,
    hole_area_threshold=50,
    close_kernel_size=3
):
    """
    对输入 mask 进行以下操作：
    1) 去除小块噪声
    2) 填补小孔洞
    3) 形态学闭运算(可选)
    4) 连通分量分析(保留最大区域)
    5) PCA 计算主方向，输出角度
    6) 可视化全部过程
    
    参数:
    mask               : 二值掩膜(H, W)，值域可 0/1 或 0/255
    min_obj_size       : remove_small_objects 的阈值，去除小面积块
    hole_area_threshold: remove_small_holes 的面积阈值，填充小孔洞
    close_kernel_size  : 闭运算(kernel)大小，用于填补较大裂缝；若<=1则跳过

    返回:
    angle_degrees : 分子相对于 x 轴的主方向角度(度)，范围 [0, 180)。若无法计算则返回 None。
    """

    # ---- Step 0. 将 mask 转成 0/1
    if mask.max() > 1:
        bin_mask = (mask > 0).astype(np.uint8)
    else:
        bin_mask = mask.astype(np.uint8)

    # 为了可视化，先把原始 mask 备份一下
    raw_mask = bin_mask.copy()

    # ---- Step 1. 转布尔类型用于 skimage
    bool_mask = (bin_mask > 0)

    # 1.1 去除小块噪声
    bool_removed_small = remove_small_objects(bool_mask, min_size=min_obj_size)

    # 1.2 填补小孔洞
    bool_filled_holes = remove_small_holes(bool_removed_small, area_threshold=hole_area_threshold)

    # 转回 0/1
    pre_mask = bool_filled_holes.astype(np.uint8)

    # 1.3 形态学闭运算(可选)
    if close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        pre_mask_closed = cv2.morphologyEx(pre_mask, cv2.MORPH_CLOSE, kernel)
    else:
        pre_mask_closed = pre_mask.copy()

    # ---- Step 2. 连通分量分析，保留最大连通分量
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pre_mask_closed, connectivity=8)
    if num_labels <= 1:
        # 只检测到背景，没有前景
        largest_mask = np.zeros_like(pre_mask_closed)
        angle_degrees = None
    else:
        # 找到面积最大的连通分量
        max_area = 0
        max_label = 0
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = label_id

        largest_mask = (labels == max_label).astype(np.uint8)


        angle = calculate_orientation(largest_mask)

        # ---- Step 3. PCA 计算主方向
        coords = np.column_stack(np.where(largest_mask > 0))  # (N,2)，(y, x)
        if len(coords) < 2:
            angle_degrees = None
        else:
            # 质心
            y_mean, x_mean = np.mean(coords, axis=0)
            # 中心化
            coords_centered = coords - [y_mean, x_mean]
            # 协方差矩阵 + 特征分解
            cov = np.cov(coords_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            # 排序
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]

            # 第一个主成分向量
            principal_vec = eigenvectors[:, 0]  # (vy, vx)
            # 与 x 轴(列方向)的夹角
            angle_radians = np.arctan2(principal_vec[0], principal_vec[1])
            angle_degrees = np.degrees(angle_radians)
            if angle_degrees < 0:
                angle_degrees += 180.0

    # ========== 可视化各步骤结果 ==========
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    # (0) 原始 mask
    axes[0].imshow(raw_mask, cmap='gray')
    axes[0].set_title('0) Raw Mask')
    axes[0].axis('off')

    # (1) 去除小块噪声后
    axes[1].imshow(bool_removed_small, cmap='gray')
    axes[1].set_title('1) Remove Small Objects')
    axes[1].axis('off')

    # (2) 填补小孔洞后
    axes[2].imshow(bool_filled_holes, cmap='gray')
    axes[2].set_title('2) Fill Small Holes')
    axes[2].axis('off')

    # (3) 闭运算后
    axes[3].imshow(pre_mask_closed, cmap='gray')
    axes[3].set_title('3) Closing')
    axes[3].axis('off')

    # (4) 最大连通分量
    axes[4].imshow(largest_mask, cmap='gray')
    axes[4].set_title('4) Largest Component')
    axes[4].axis('off')

    # (5) 最终 PCA 方向可视化
    axes[5].imshow(largest_mask, cmap='gray')
    axes[5].set_title('5) Final Orientation')
    axes[5].axis('off')

    if angle_degrees is not None:
        # 在图上画出 PCA 主轴箭头
        coords = np.column_stack(np.where(largest_mask > 0))
        y_mean, x_mean = np.mean(coords, axis=0)  # 中心
        # principal_vec 的计算再来一次（也可在外面缓存，这里为了可视化清晰）
        coords_centered = coords - [y_mean, x_mean]
        cov = np.cov(coords_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        principal_vec = eigenvectors[:, 0]  # (vy, vx)

        # 适当放大箭头长度
        arrow_scale = 0.2 * max(mask.shape[0], mask.shape[1])  
        vy, vx = principal_vec
        x2 = x_mean + arrow_scale * vx
        y2 = y_mean + arrow_scale * vy

        axes[5].arrow(x_mean, y_mean, (x2 - x_mean), (y2 - y_mean),
                      color='red', head_width=5, length_includes_head=True)
        # 画质心
        axes[5].scatter(x_mean, y_mean, s=30, c='red', marker='x')
        # 在标题上标注角度
        axes[5].set_title(f'Angle = {angle_degrees:.2f}°')
    else:
        axes[5].set_title('No valid orientation')

    plt.tight_layout()
    plt.show()

    return angle_degrees

# =======================
# 使用示例
if __name__ == "__main__":
    # # 构造一个带孔洞 + 噪声的示例 mask
    # H, W = 200, 200
    # mask_demo = np.zeros((H, W), dtype=np.uint8)

    # # 在靠近中心位置画一个方块
    # pts = np.array([
    #     [70, 70],
    #     [70, 130],
    #     [130, 130],
    #     [130, 70]
    # ], dtype=np.int32)
    # # 旋转
    # rot_mat = cv2.getRotationMatrix2D((100, 100), 45, 1.0)
    # pts_rot = cv2.transform(np.array([pts]), rot_mat)[0]
    # cv2.fillConvexPoly(mask_demo, pts_rot.astype(np.int32), 255)

    # # 在里面弄个小孔洞
    # cv2.circle(mask_demo, (100, 100), 5, 0, -1)

    # # 随机添加一些噪声点
    # for _ in range(80):
    #     x_noise = np.random.randint(0, W)
    #     y_noise = np.random.randint(0, H)
    #     mask_demo[y_noise, x_noise] = 255  # 随意点一些

    image_path = './mol_segment/058_test/367_5.png'
    output_folder = './mol_segment/results1'
    model_path = './mol_segment/unet_model-zzw-Ni.pth'
    img = Image.open(image_path).convert('RGB')
    _,_,mask = segmented_image(img, output_folder,model_path)



    if len(mask.shape) == 3:
        mask = mask.squeeze().astype(np.uint8)
    # resize the mask to (208,208)
    mask = cv2.resize(mask, (208, 208), interpolation=cv2.INTER_NEAREST)
    angle = estimate_and_visualize(
        mask,
        min_obj_size=100,
        hole_area_threshold=500,
        close_kernel_size=20
    )
    print(f"Estimated orientation angle = {angle}")
