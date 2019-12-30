#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/20 14:57
# @Author  : Guicheng.Zhou
# @File    : faceswap.py
# @Software: IntelliJ IDEA
# @Describe: 人脸交换

import dlib
import numpy as np
import cv2
from FaceSwap.ExceptionClass import *

# 加载训练模型
PREDICTOR_PATH = 'data\shape_predictor_68_face_landmarks.dat'
# 图像缩放因子
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

# 根据draw_landmarks.py运行结果,分析标记图如下：
JAW_POINTS = list(range(0, 17))  # 下巴
RIGHT_BROW_POINTS = list(range(17, 22))  # 右眉毛
LEFT_BROW_POINTS = list(range(22, 27))  # 左眉毛
RIGHT_EYE_POINTS = list(range(36, 42))  # 右眼睛
LEFT_EYE_POINTS = list(range(42, 48))  # 左眼睛
NOSE_POINTS = list(range(27, 36))  # 鼻子
MOUTH_POINTS = list(range(48, 68))  # 嘴巴
FACE_POINTS = list(range(17, 68))  # 脸

# 选取左右眉毛、眼睛、鼻子、嘴巴的位置作为特征点索引
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_BROW_POINTS + LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_POINTS + MOUTH_POINTS)

# 选取用于叠加在第一张脸上的第二张脸的面部特征，特征点包括左右眼、眉毛、鼻子、嘴巴
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS]

# 定义用于颜色校正的模糊量，作为瞳孔距离的系数
COLOUR_CORRECT_BLUR_FRAC = 0.6

# 实例化脸部检测器
detector = dlib.get_frontal_face_detector()
# 加载训练模型并实例化特征提取器
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def read_img_and_landmarks(filename):
    """
    读取图片并标记
    :param filename:文件名(含路径)
    :return: im对象和图像特征点
    """
    # 以RGB模式读取图像
    im = cv2.imread(filename, cv2.IMREAD_COLOR)
    # 对图像进行适当的缩放
    '''
    hc.jpg 1200*1198
    print(im.shape)  #显示尺寸 (1198, 1200, 3)
    print(im.shape[0])  #图片高度  1198
    print(im.shape[1])  #图片宽度  1200
    print(im.shape[2])  #图片通道数    3
    '''
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s


def get_landmarks(im):
    """
    获取特征点
    :param im:im对象
    :return: 图像特征点，2维矩阵
    """
    # detector是特征检测器
    # predictor是特征提取器
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    # 返回一个n*2维的矩阵，该矩阵由检测到的脸部特征点坐标组成
    '''
    numpy:array数组 matrix矩阵
    '''
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def transformation_from_points(points1, points2):
    """
    计算转换信息，返回变换矩阵
    :param points1: 图像特征点矩阵1
    :param points2: 图像特征点矩阵2
    :return: 变换矩阵
    """
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    '''
    mean()函数功能：求取均值
    经常操作的参数为axis，以m * n矩阵举例：
    axis 不设置值，对 m*n 个数求均值，返回一个实数
    axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    '''
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    '''
    std()计算矩阵标准差
    np.std(a) 计算全局标准差
    np.std(a,axis=0) 计算每一列的标准差
    np.std(a,axis=1) 计算每一行的标准差
    '''
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def get_face_mask(im, landmarks):
    """
    获取面部的掩码
    :param im: im对象
    :param landmarks:特征点
    :return: im
    """
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))
    # 引用高斯模糊
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im


def draw_convex_hull(im, points, color):
    """
    绘制凸多边形
    :param im: im对象
    :param points: 特征点
    :param color: 颜色
    :return:
    """
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def warp_im(im, M, dshape):
    """
    变换图像
    :param im:
    :param M:
    :param dshape:
    :return:
    """
    output_im = np.zeros(dshape, dtype=im.dtype)
    # 仿射函数,能对图像进行几何变换
    # 三个主要参数,第一个输入图像，第二个变化矩阵np.float32类型,第三个变换之后图像的宽高
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst=output_im, borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    """
    修正颜色
    :param im1:
    :param im2:
    :param landmarks1:
    :return:
    """
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    # 避免出现0除
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)


if __name__ == '__main__':
    img1 = 'data\dt.jpg'
    img2 = 'data\hc.jpg'

    # 获取图像与特征点
    im1, landmarks1 = read_img_and_landmarks(img1)
    im2, landmarks2 = read_img_and_landmarks(img2)

    # 选取两组图像特征矩阵中所需要的面部部位，计算转换信息，返回变换矩阵
    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

    # 获取im2的面部掩码
    mask = get_face_mask(im2, landmarks2)
    # 将im2的掩码进行变化,使之与im1相符
    warped_mask = warp_im(mask, M, im1.shape)
    # 将二者的掩码进行连通
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)
    # 将第二幅图像调整到第一幅图像相符
    warped_im2 = warp_im(im2, M, im1.shape)
    # 将im2的皮肤颜色进行修正,使其和im1的颜色尽量协调
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    # 组合图像，获得结果
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    # 保存图像
    cv2.imwrite('data\output.jpg', output_im)
