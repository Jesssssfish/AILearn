#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/2 16:22
# @Author  : Guicheng.Zhou
# @File    : load_face_dataset41.py
# @Software: IntelliJ IDEA
# @Describe: 加载人脸数据集

import cv2
import os
import numpy as np

IMAGE_SIZE = 64


def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    """
    调整图像大小
    :param image:图像
    :param height: 高
    :param width: 宽
    :return:
    """
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 对于长宽不等的图片，找到最长的一边
    longest_edge = max(h, w)
    # 计算短边需要增加多少像素等于长边
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2  # 取整除
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # RGB颜色
    BLACK = [0, 0, 0]
    # 给图像增加边界，使长宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))


# 读取训练数据
images = []
labels = []


def read_path(path_name):
    """
    读取图片
    :param path_name:路径
    :return: 图片对象集合，图片对象路径文本集合
    """
    for dir_item in os.listdir(path_name):
        # 从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是文件夹继续递归
            read_path(full_path)
        else:  # 文件
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                # image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                labels.append(path_name)
    return images, labels


# 从指定路径读取训练数据
def load_dataset(path_name):
    images, labels = read_path(path_name)

    # 将输入的所有图片转成四维数组，尺寸为（图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    # 图片为64*64,一个像素3个颜色值(RGB)
    images = np.array(images)
    nlabels = []
    # 标注数据
    for label in labels:
        if 'zgc' in label:
            nlabels.append(0)
        elif 'zjq' in label:
            nlabels.append(1)
    return images, nlabels


if __name__ == '__main__':
    load_dataset('F:/AI/videos/face')
