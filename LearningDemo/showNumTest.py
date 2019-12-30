#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/9 14:07
# @Author  : Guicheng.Zhou
# @File    : showNumTest.py
# @Software: IntelliJ IDEA
# @Describe: 显示数字,把数字的像素数据转换成图片形式显示

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_file = open('data/mnist_train_100.csv', 'r')
    data_list = data_file.readlines()
    print(len(data_list))

    all_values = data_list[1].split(',')
    image_array = np.asfarray(all_values[1:]).reshape(28, 28)
    print(all_values[1])
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()
    data_file.close()
