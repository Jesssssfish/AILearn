#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/13 16:42
# @Author  : Guicheng.Zhou
# @File    : SinglePerceptron.py
# @Software: IntelliJ IDEA
# @Describe: 单层感知器

"""
假设平面上有三个点,(3,3),(4,3)这两个点标签为1，(1,1)这个点标签为-1，构建神经网络分类
"""

import numpy as np
import matplotlib.pyplot as plt

# 输入数据
X = np.array([[1, 3, 3], [1, 4, 3], [1, 1, 1]])
# 标签，期望输出
Y = np.array([1, 1, -1])
# 权值初始化，1行3列的矩阵，取值范围是-1到1
W = (np.random.random(3) - 0.5) * 2  # (-0.5,0.2,0.9)
print('随机权值：', W)
# 学习率设置
lr = 0.11
# 计算迭代次数
n = 0
# 神经网路输出
O = 0


def update():
    """
    更新权值函数
    :return:
    """
    global X, Y, W, lr, n
    n += 1
    O = np.sign(np.dot(X, W.T))
    W_C = lr * ((Y - O.T).dot(X) / int(X.shape[0]))
    W = W + W_C


for _ in range(100):
    update()
    print('更新后的权值:', W)
    print('当前迭代次数：', n)
    O = np.sign(np.dot(X, W.T))
    if (O == Y.T).all():
        print("Finished")
        print("epoch:", n)
        break

# 正样本
x1 = [3, 4]
y1 = [3, 3]

# 负样本
x2 = [1]
y2 = [1]

# 计算分界线斜率以及截距 y=kx+d
k = -W[1] / W[2]
d = -W[0] / W[2]
print('k=', k)
print('d=', d)

xdata = np.linspace(0, 5)

plt.figure()
plt.plot(xdata, xdata * k + d, 'r')
plt.plot(x1, y1, 'bo')
plt.plot(x2, y2, 'yo')
plt.show()
