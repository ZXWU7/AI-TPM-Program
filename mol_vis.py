from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# draw a equilateral triangle in edge length 100, with the center at (0, 0)
def triangle_norm(position = (0,0) ,value_points=(250,250), size=100, key_points_size_factor=0.6, center_std_devs = 18, center_weights = 1, edge_std_devs = 15, edge_weights = 0.6):

    cx, cy = position
    # the three vertexes of the triangle
    vertexes = np.array([[cx, cy - sqrt(3) / 3 * size], 
                         [cx - size / 2, cy + sqrt(3) / 6 * size], 
                         [cx + size / 2, cy + sqrt(3) / 6 * size], 
                         [cx, cy - sqrt(3) / 3 * size]])
    # draw the four key points of the triangle and the center, the key points are 0.8 times the edge length in sdie of the vertexes
    key_points = np.array([[cx, cy - sqrt(3) / 3 * size * key_points_size_factor], 
                           [cx - size / 2 * key_points_size_factor, cy + sqrt(3) / 6 * size * key_points_size_factor], 
                           [cx + size / 2 * key_points_size_factor, cy + sqrt(3) / 6 * size * key_points_size_factor], 
                           [cx, cy]])
    
    centers = key_points
    means = ([0, 0], [0, 0], [0, 0], [0, 0])
    std_devs = ([edge_std_devs, edge_std_devs], [edge_std_devs, edge_std_devs], [edge_std_devs, edge_std_devs], [center_std_devs, center_std_devs])
    weights = [edge_weights, edge_weights, edge_weights, center_weights]

    x = np.linspace(-size*2, size*2, size*4)
    y = x
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    value = 0

        # 计算每个正态分布并叠加
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
            -(((value_points[0] - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
              ((value_points[1] - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
        )
    # normalize the Z to 0-1
    nor_Z = Z / np.max(Z)
    # Z = Z / np.max(Z)
    value = value / np.max(Z)

    return value, (X, Y, nor_Z), vertexes

def triangle_norm_distribution(position = (0,0) ,value_points=(250,250), size=100, key_points_size_factor=0.6, center_std_devs = 18, center_weights = 1, edge_std_devs = 16, edge_weights = 0.5):

    cx, cy = position
    # the three vertexes of the triangle
    # draw the four key points of the triangle and the center, the key points are 0.8 times the edge length in sdie of the vertexes
    key_points = np.array([[cx, cy - sqrt(3) / 3 * size * key_points_size_factor], 
                           [cx - size / 2 * key_points_size_factor, cy + sqrt(3) / 6 * size * key_points_size_factor], 
                           [cx + size / 2 * key_points_size_factor, cy + sqrt(3) / 6 * size * key_points_size_factor], 
                           [cx, cy]])
    
    centers = key_points
    means = ([0, 0], [0, 0], [0, 0], [0, 0])
    std_devs = ([edge_std_devs, edge_std_devs], [edge_std_devs, edge_std_devs], [edge_std_devs, edge_std_devs], [center_std_devs, center_std_devs])
    weights = [edge_weights, edge_weights, edge_weights, center_weights]

    x = np.linspace(-size*2, size*2, size*4)
    y = x
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    value = 0

        # 计算每个正态分布并叠加
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
            -(((value_points[0] - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
              ((value_points[1] - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
        )
    # Z = Z / np.max(Z)
    value = value / np.max(Z)

    return value


def square_norm(position = (0,0) ,value_points=(250,250), size=100, key_points_size_factor=0.6, center_std_devs = 18, center_weights = 1, edge_std_devs = 15, edge_weights = 0.6):

    cx, cy = position
    # the four vertexes of the square
    vertexes = np.array([   [cx + size / 2 , cy ], 
                            [cx , cy + size / 2 ], 
                            [cx - size / 2 , cy ], 
                            [cx , cy - size / 2 ],
                            [cx + size / 2 , cy ]])
    # draw the four key points of the square and the center, the key points are 0.8 times the edge length in sdie of the vertexes
    key_points = np.array([[cx + size / 2 , cy ], 
                           [cx , cy + size / 2 ], 
                           [cx - size / 2 , cy ], 
                           [cx , cy - size / 2 ], 
                           [cx, cy]])
    
    centers = key_points
    means = ([0, 0], [0, 0], [0, 0], [0, 0], [0, 0])
    std_devs = ([edge_std_devs, edge_std_devs], [edge_std_devs, edge_std_devs], [edge_std_devs, edge_std_devs], [edge_std_devs, edge_std_devs], [center_std_devs, center_std_devs])
    weights = [edge_weights, edge_weights, edge_weights, edge_weights, center_weights]

    x = np.linspace(-size*2, size*2, size*4)
    y = x
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    value = 0

        # 计算每个正态分布并叠加
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
            -(((value_points[0] - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
              ((value_points[1] - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
        )
    # normalize the Z to 0-1
    nor_Z = Z / np.max(Z)
    # Z = Z / np.max(Z)
    value = value / np.max(Z)
    
    return value, (X, Y, nor_Z), vertexes

# a quarter of the square norm distribution
def petal_norm(position = (0,0) ,value_points=(250,250), petal_site = 0 , size=100, key_points_size_factor=0.6, center_std_devs = 18, center_weights = 1, edge_std_devs = 15, edge_weights = 0.6):

    cx, cy = position
    # the four vertexes of the square
    vertexes = np.array([   [cx - size / 2 , cy - size / 2], 
                           [cx + size / 2 , cy - size / 2 ], 
                           [cx + size / 2 , cy + size / 2 ], 
                           [cx - size / 2, cy + size / 2 ],
                           [cx - size / 2 , cy - size / 2]])
    # draw the two key points of the petal and the center
    petal_site_list = [[cx - size / 2 , cy - size / 2], 
                           [cx + size / 2 , cy - size / 2 ], 
                           [cx + size / 2 , cy + size / 2 ], 
                           [cx - size / 2, cy + size / 2 ]]
    key_points = np.array([petal_site_list[petal_site], [cx, cy]])

    centers = key_points
    means = ([0, 0], [0, 0])
    std_devs = ([edge_std_devs, edge_std_devs], [center_std_devs, center_std_devs])
    weights = [edge_weights, center_weights]

    x = np.linspace(-size*2, size*2, size*4)
    y = x
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    value = 0

        # 计算每个正态分布并叠加    
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
            -(((value_points[0] - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
              ((value_points[1] - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
        )
    # normalize the Z to 0-1
    nor_Z = Z / np.max(Z)
    # Z = Z / np.max(Z)
    value = value / np.max(Z)

    return value, (X, Y, nor_Z), vertexes

if __name__ == '__main__':
    
    triangle_position_list = [(100, 100), (250, 250), (400, 400)]
    size = 160

    plt.figure(figsize=(13, 10))
    x = np.linspace(-size*2, size*2, size*4)
    y = x
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for triangle_position in triangle_position_list:
        
        # value, data4draw, vertexes = triangle_norm(value_points=(0,10))
        # value, data4draw, vertexes = square_norm(value_points=(0,10))
        dis_args = (50, 1, 45, 0.6)
        value, data4draw, vertexes = petal_norm(size=size,value_points=(0,10),petal_site=1, center_std_devs = dis_args[0], center_weights = dis_args[1], edge_std_devs = dis_args[2], edge_weights = dis_args[3])
        X, Y, nor_Z = data4draw
        Z += nor_Z
        # draw the triangle
        plt.plot(vertexes[:, 0], vertexes[:, 1], color='black')
    
    # normalize the Z to 0-0.8
    Z = Z / np.max(Z)
    # draw the hot map
    contour = plt.contourf(X, Y, Z, levels=30, cmap='viridis')    
    plt.colorbar(contour, label="Probability Density")    
    plt.title("Multivariate Normal Distributions", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

    # value = triangle_norm_distribution(value_points=(0,10))
    # print(value)
