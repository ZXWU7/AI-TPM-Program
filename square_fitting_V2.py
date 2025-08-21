import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects


def create_random_square_mask(H=200, W=200, side=50, angle = 30):
    """
    生成一个随机旋转的近似正方形 mask
    :param H: mask 高度
    :param W: mask 宽度
    :param side: 正方形边长（像素）
    :return: 二值 mask, 以及该正方形的实际四个顶点坐标(旋转后)
    """
    # 随机一个中心点，保证方块不会超出边界
    center_x = np.random.randint(side, W - side)
    center_y = np.random.randint(side, H - side)
    
    # 随机一个旋转角度
    angle = angle # np.random.uniform(0, 180)

    # 构建正方形四个顶点（未旋转前）
    # 注意：OpenCV 中图像坐标 (x, y) 往往指 (列, 行)，
    # 但这里为了直观，先用 (cx±, cy±) 这种传统 (x, y) 表达。
    half_side = side / 2.0
    pts = np.array([
        [center_x - half_side, center_y - half_side],
        [center_x + half_side, center_y - half_side],
        [center_x + half_side, center_y + half_side],
        [center_x - half_side, center_y + half_side]
    ], dtype=np.float32)

    # 构建旋转矩阵 (OpenCV 里第一个参数是旋转中心, 后面是旋转角和缩放)
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # 对顶点进行旋转变换
    pts_rot = cv2.transform(np.array([pts]), rot_mat)[0]

    # 创建空白 mask，填充旋转后的正方形
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_rot.astype(np.int32), 1)

    return mask, pts_rot

def pca_analysis(mask):
    """
    对 mask 进行 PCA，返回：质心、特征向量、特征值
    """
    # 提取所有前景像素坐标
    coords = np.column_stack(np.where(mask > 0))  # (N,2)，每行是 (y,x)
    if len(coords) < 2:
        # 如果连两个像素都没有，则返回空
        return (0,0), None, None
    
    # 计算质心
    y_mean, x_mean = np.mean(coords, axis=0)
    center = (x_mean, y_mean)
    
    # 中心化坐标
    coords_centered = coords - [y_mean, x_mean]

    # 计算协方差矩阵
    cov = np.cov(coords_centered, rowvar=False)
    # 求特征值与特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # 按特征值从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return center, eigenvectors, eigenvalues

def preprocess_and_pca(mask):
    # 1) 转布尔，并去除小噪声
    bin_mask = (mask > 0)
    bin_mask = remove_small_objects(bin_mask, min_size=50)  # 根据实际情况设置

    # 2) 填补小孔洞
    bin_mask = remove_small_holes(bin_mask, area_threshold=50)

    # 3) 转回 0/1
    clean_mask = bin_mask.astype(np.uint8)

    # 4) (可选)再用形态学闭运算，填补较大的缝隙
    kernel = np.ones((3,3), np.uint8)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # 5) 连通分量分析，选出面积最大的那个
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    if num_labels <= 1:
        # 没有前景
        return None, None, None
    
    # 找最大分量
    max_area = 0
    max_label = 0
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = label_id

    final_mask = (labels == max_label).astype(np.uint8)

    # 6) 在此基础上做 PCA
    coords = np.column_stack(np.where(final_mask > 0))  # (N,2)
    if len(coords) < 2:
        return None, None, None

    # 计算质心
    y_mean, x_mean = np.mean(coords, axis=0)
    center = (x_mean, y_mean)

    # 中心化
    coords_centered = coords - [y_mean, x_mean]
    cov = np.cov(coords_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return final_mask, center, (eigenvalues, eigenvectors)

def draw_aligned_square(image, center, angle_degrees, side_length=10, color=(0,255,0), thickness=2):
    """
    在图像上以 center 为中心，旋转 angle_degrees 度，绘制一个边长为 side_length 的正方形。
    :param image: 原图 (彩色 或 灰度)
    :param center: (cx, cy)
    :param angle_degrees: 旋转角度
    :param side_length: 正方形边长
    :param color: 绘制颜色 (B, G, R)
    :param thickness: 线宽
    :return: 绘制后的图像
    """
    (cx, cy) = center

    # 正方形的一半边长
    half_side = side_length / 2
    # 以 center 为中心，构建正方形四个角点（未旋转前）
    pts = np.array([
        [cx - half_side, cy - half_side],
        [cx + half_side, cy - half_side],
        [cx + half_side, cy + half_side],
        [cx - half_side, cy + half_side]
    ], dtype=np.float32)

    # 构建旋转矩阵：注意 OpenCV 中图像坐标 (cx, cy) -> (列, 行)
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_degrees, 1.0)

    # 旋转四个顶点
    pts_rot = cv2.transform(np.array([pts]), rot_mat)[0]

    # 转换为 int 并绘制
    pts_int = pts_rot.astype(np.int32)
    cv2.polylines(image, [pts_int], isClosed=True, color=color, thickness=thickness)

    return image, pts_rot

def visualize_pca(mask, pts_rot, center, eigenvectors, eigenvalues):

    """
    使用 matplotlib 进行可视化：
    1) 二值 mask 的散点图
    2) 质心
    3) PCA 主轴
    """
    # 得到当前时间，用于保存图片
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    coords = np.column_stack(np.where(mask > 0))  # (N,2) -> (y,x)

    plt.figure(figsize=(6,6))
    plt.title("Random Square Mask + PCA")

    # -- 1) 绘制 mask 的像素散点
    plt.scatter(coords[:,1], coords[:,0], c='red', s=5, alpha=0.5, label='Mask Pixels')

    # -- 2) 绘制正方形顶点连线（旋转后的形状）
    # 注意：pts_rot 里面是 (x, y)；coords 里面是 (y, x)
    # 为了和上面散点顺序一致，需要调换一下顺序
    pts_plot = pts_rot[:, [0,1]]
    pts_plot = np.vstack([pts_plot, pts_plot[0]])  # 闭合
    plt.plot(pts_plot[:,0], pts_plot[:,1], c='green', label='Rotated Square')

    # -- 3) 绘制质心
    cx, cy = center
    plt.scatter(cx, cy, c='blue', s=50, marker='x', label='Center')

    # -- 4) 绘制 PCA 主轴
    # eigenvectors 形状是 (2,2)，列是第 i 个特征向量
    # 我们先将它们都可视化出来，长度与特征值相关
    # 为了可视化，适当乘以一个放大系数
    scale = 0.5
    for i in range(2):
        # 主成分向量 (vy, vx)
        vx = eigenvectors[1, i]
        vy = eigenvectors[0, i]
        val = eigenvalues[i]

        # 向量起点(cx, cy)，终点 = 起点 + scale * sqrt(val) * (vx, vy)
        x2 = cx + scale * np.sqrt(val) * vx
        y2 = cy + scale * np.sqrt(val) * vy

        plt.arrow(cx, cy, (x2 - cx), (y2 - cy),
                  head_width=2, head_length=4,
                  fc=('magenta' if i==0 else 'cyan'), 
                  ec=('magenta' if i==0 else 'cyan'),
                  length_includes_head=True)
        plt.text(x2, y2,
                 f'PC{i+1}',
                 color=('magenta' if i==0 else 'cyan'),
                 fontsize=10)

    # 调整坐标轴和显示
    plt.gca().invert_yaxis()  # 图像坐标y轴向下
    plt.axis('equal')
    plt.legend()
    # 如果save_path不存在，创建它
    save_path = f"./pca_analysis/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"./pca_analysis/pca_analysis_{now}.png")
    # plt.show()

if __name__ == "__main__":
    angle_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # 1. 随机生成一个近似正方形 mask H和W也是随机的(150~)
    for angle in angle_list:
        time.sleep(1)
        mask, pts_rot = create_random_square_mask(H=200, W=200, side=50, angle=angle)

        # 2. 对该 mask 做 PCA
        center, eigenvectors, eigenvalues = pca_analysis(mask)

        # 3. 可视化
        if eigenvectors is not None:
            visualize_pca(mask, pts_rot, center, eigenvectors, eigenvalues)
        else:
            print("Mask 太小，无法执行 PCA。")
