
# import matplotlib.pyplot as plt
# import numpy as np


# def normal_distribution(x, mean, std_dev):
#     return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)/(1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((0 - mean) / std_dev) ** 2)

# def ind_distribution(x):
#     return (1-normal_distribution(x, 0, 1) if x >0 else 0) + normal_distribution(x, 0, 1)*0.02

# def non_distribution(x):
#     return (1-normal_distribution(x, 0, 1) if x <0 else 0) + normal_distribution(x, 0, 1)*0.02

# def suc_distribution(x):
#     return 1 - ind_distribution(x) - non_distribution(x)

# x = np.linspace(-10, 10, 500)
# ind_list = []
# non_list = []
# suc_list = []

# for i in x:
#     ind_list.append(ind_distribution(i))
#     non_list.append(non_distribution(i))
#     suc_list.append(suc_distribution(i))
    

# # plot
# plt.figure(figsize=(10, 6))
# plt.plot(x, suc_list, label="suc Distribution", linewidth=2)
# # plt.plot(x, sigmoid_y_convert, label="Sigmoid Convert", linewidth=2)
# plt.plot(x, ind_list, label="ind Distribution", linewidth=2)
# plt.plot(x, non_list, label="non Distribution", linewidth=2)
# plt.title("Sigmoid Function and Normal Distribution", fontsize=14)
# plt.xlabel("X", fontsize=12)
# plt.ylabel("Y", fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
# plt.show()


import matplotlib.pyplot as plt
# 重新导入必要的库
import numpy as np


# 定义二维正态分布
def normal_distribution_2d(x, y, mean_x, mean_y, std_dev_x, std_dev_y):
    """
    二维正态分布概率密度函数，归一化到(0, mean)为1
    """
    norm_factor = (1 / (std_dev_x * std_dev_y * 2 * np.pi))
    exponent = -0.5 * (((x - mean_x) / std_dev_x) ** 2 + ((y - mean_y) / std_dev_y) ** 2)
    normal_value = norm_factor * np.exp(exponent)
    normalization = norm_factor * np.exp(-0.5 * ((0 - mean_x) / std_dev_x) ** 2 + ((0 - mean_y) / std_dev_y) ** 2)
    return normal_value / normalization

# 定义ind_distribution, non_distribution和suc_distribution在二维上的公式
def non_distribution_2d(x, y):
    if x<=0 and y>=0:
        return 1 - normal_distribution_2d(x, 0, 0, 0, 1, 1)+normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02
    if x<0 and y<0:
        return (1 - normal_distribution_2d(x, 0, 0, 0, 1, 1))+normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02
    # if x<0 and y>0:
    #     return (1 - normal_distribution_2d(0, y, 0, 0, 1, 1))+normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02
    else:
        return normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02 


def suc_distribution_2d(x, y):
    return normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.96

def ind_distribution_2d(x, y):
    if x>=0 and y<=0:
        return 1 - normal_distribution_2d(x, 0, 0, 0, 1, 1) + normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02
    if x<0 and y<0:
        return normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02

    return 1-non_distribution_2d(x, y) - suc_distribution_2d(x, y)

def non_distribution_2d_V2(x, y):
        return 1-suc_distribution_2d(x, y) - ind_distribution_2d(x, y)


# 生成网格数据
grid_size = 100
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)

# 计算分布
Z_ind = np.vectorize(ind_distribution_2d)(X, Y)
Z_non = np.vectorize(non_distribution_2d_V2)(X, Y)
Z_suc = np.vectorize(suc_distribution_2d)(X, Y)

ind_x = np.zeros_like(x)
suc_x = np.zeros_like(x)
non_x = np.zeros_like(x)
for i in range(len(x)):
    ind_x[i] = ind_distribution_2d(x[i], 0)
    suc_x[i] = suc_distribution_2d(x[i], 0)
    non_x[i] = non_distribution_2d_V2(x[i], 0)

ind_y = np.zeros_like(y)
suc_y = np.zeros_like(y)
non_y = np.zeros_like(y)
for i in range(len(y)):
    ind_y[i] = ind_distribution_2d(0, y[i])
    suc_y[i] = suc_distribution_2d(0, y[i])
    non_y[i] = non_distribution_2d_V2(0, y[i])


# 绘制等高线图
plt.figure(figsize=(15, 8))

# ind_distribution
plt.subplot(2, 3, 1)
plt.contourf(X, Y, Z_ind, levels=30, cmap="Reds")
plt.colorbar(label="ind_distribution")
plt.title("ind_distribution")
plt.xlabel("X")
plt.ylabel("Y")

# non_distribution
plt.subplot(2, 3, 2)
plt.contourf(X, Y, Z_non, levels=30, cmap="Blues")
plt.colorbar(label="non_distribution")
plt.title("non_distribution")
plt.xlabel("X")
plt.ylabel("Y")

# suc_distribution
plt.subplot(2, 3, 3)
plt.contourf(X, Y, Z_suc, levels=30, cmap="Greens")
plt.colorbar(label="suc_distribution")
plt.title("suc_distribution")
plt.xlabel("X")
plt.ylabel("Y")

# x axis
plt.subplot(2, 3, 4)
plt.plot(x, ind_x, color="black", linewidth=2)
plt.plot(x, suc_x, color="green", linewidth=2)
plt.plot(x, non_x, color="blue", linewidth=2)
plt.title("distribution on X axis")
plt.xlabel("X")
plt.ylabel("ind_distribution")

# y axis
plt.subplot(2, 3, 5)
plt.plot(y, ind_y, color="black", linewidth=2)
plt.plot(y, suc_y, color="green", linewidth=2)
plt.plot(y, non_y, color="blue", linewidth=2)
plt.title("ind_distribution on Y axis")
plt.xlabel("Y")
plt.ylabel("distribution")

plt.tight_layout()
plt.show()
