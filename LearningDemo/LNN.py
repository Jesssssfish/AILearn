#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/16 15:26
# @Author  : Guicheng.Zhou
# @File    : LNN.py
# @Software: IntelliJ IDEA
# @Describe: LNN(线性神经网络,Linear neural network),解决单层感知器异或问题(加入非线性输入，使等效输入维度变得更大)

"""
假设平面上有四个点(0,0),(0,1),(1,0),(1,1),(0,0)和(1,1)归为一类,用-1表示
(0,1)和(1,0)归为一类,用1表示

"""
import numpy as np
import matplotlib.pyplot as plt

# 输入数据
X = np.array([[1, 0, 0, 0, 0, 0],  # x0=0,x1=0,x2=0,x3=x1x1,x4=x1*x2,x5=x2*x2,加入非线性输入
              [1, 0, 1, 0, 0, 1],
              [1, 1, 0, 1, 0, 0],
              [1, 1, 1, 1, 1, 1]])
# 标签,期望输出
Y = np.array([-1, 1, 1, - 1])
# 权值初始化,1行6列的矩阵，取值范围-1到1
W = (np.random.random(6) - 0.5) * 2

print("随机权值:", W)

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
    O = np.dot(X, W.T)  # 线性函数y=x
    W_C = lr * ((Y - O.T).dot(X) / int(X.shape[0]))
    W = W + W_C


for _ in range(1500):
    update()
    # 收敛有三种方式：
    # 1：误差达到比较小的值,如单层感知器里用的
    # 2:权值改变量较小的时候
    # 3:循环一定的次数,这里使用这种方式

# 正样本
x1 = [0, 1]
y1 = [1, 0]

# 负样本
x2 = [0, 1]
y2 = [0, 1]


def calculate(x, root):
    """
    计算函数，画线
    :param x:
    :param root:
    :return:
    """
    a = W[5]
    b = W[2] + x * W[4]
    c = W[0] + W[1] * x + W[3] * x * x
    if root == 1:
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)  # sqrt开方
    if root == 2:
        return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


xdata = np.linspace(-1, 2)

plt.figure()

plt.plot(xdata, calculate(xdata, 1), 'r')
plt.plot(xdata, calculate(xdata, 2), 'r')

plt.plot(x1, y1, 'bo')
plt.plot(x2, y2, 'yo')
plt.show()

O = np.dot(X, W.T)
print(O)  # [-0.99911601  0.99940969  0.99940969 -0.99954323]结果无限接近期望值
