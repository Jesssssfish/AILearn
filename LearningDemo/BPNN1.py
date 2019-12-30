#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/18 13:23
# @Author  : Guicheng.Zhou
# @File    : BPNN1.py
# @Software: IntelliJ IDEA
# @Describe: BP神经网络例子1：解决单层感知器异或问题，BP神经网络使用多层网络模型，输入层、隐藏层输出层

import numpy as np

# 输入数据
X = np.array([[1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

# 标签
Y = np.array([[0, 1, 1, 0]])
# 权值初始化 -1到1
V = np.random.random((3, 4)) * 2 - 1  # 3行4列
W = np.random.random((4, 1)) * 2 - 1  # 4行1列
print("V:", V)
print("W:", W)
# 学习率设置
lr = 0.11


def sigmoid(x):
    """
    激活函数
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    """
    sigmoid的导数
    :param x:
    :return:
    """
    return x * (1 - x)


def update():
    """
    更新权值的函数
    :return:
    """
    global X, Y, W, V, lr

    L1 = sigmoid(np.dot(X, V))  # 隐藏层的输出(4X4)
    L2 = sigmoid(np.dot(L1, W))  # 输出层的输出(4X1)

    L2_delta = (Y.T - L2) * dsigmoid(L2)
    L1_delta = L2_delta.dot(W.T) * dsigmoid(L1)

    W_C = lr * L1.T.dot(L2_delta)
    V_C = lr * X.T.dot(L1_delta)

    W = W + W_C
    V = V + V_C


for i in range(100000):
    update()  # 更新权值
    if i % 500 == 0:
        L1 = sigmoid(np.dot(X, V))  # 隐藏层的输出(4X4)
        L2 = sigmoid(np.dot(L1, W))  # 输出层的输出(4X1)
        print("Error=", np.mean(np.abs(Y.T - L2)))

L1 = sigmoid(np.dot(X, V))  # 隐藏层的输出(4X4)
L2 = sigmoid(np.dot(L1, W))  # 输出层的输出(4X1)
print(L2)


def judge(x):
    if x >= 0.5:
        return 1
    else:
        return 0


for i in map(judge, L2):
    print(i)
