#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 13:51
# @Author  : Guicheng.Zhou
# @File    : 3.CatchPICFromVideoUseDlib.py
# @Software: IntelliJ IDEA
# @Describe: 通过已录好的视频截取(使用Dlib)人脸并保存，用于数据训练，dlib精度较高


import cv2
import dlib
import random
from PIL import Image
import numpy as np
import os
import time
import threading

size = 64
# 加载面部检测器
detector = dlib.get_frontal_face_detector()


def relight(img, light=1, bias=0):
    """
    随机改变对比度与亮度，提高照片样本多样性
    :param img:图片
    :param light:亮度
    :param bias:对比度
    :return:图片
    """
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img


def catchpicfromvideo(video_path, file_path):
    """
    通过视频截取人脸
    :param video_path: video路径
    :param file_path: 人脸保存路径
    :return:
    """

    # 视频来源
    cap = cv2.VideoCapture(video_path)

    num = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rects, image = face_detector(frame)
        for i, d in enumerate(rects):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = image[x1:y1, x2:y2]
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            face = cv2.resize(face, (size, size))
            img_name = '%s/%d.jpg' % (file_path, num)
            num += 1
            cv2.imwrite(img_name, face)
            print(num)

    cap.release()
    cv2.destroyAllWindows()


def face_detector(img, rotate=90):
    """
    人脸检测器,
    :param img: cv2.img
    :param rotate: 逆时针旋转角度 ,默认90
    :return: 人脸结果
    """
    rotate_num = 0  # 旋转次数
    rects = []  # 人脸
    image = None  # 检测的图片
    while True:
        # 将当前帧转换成灰度图像,减少计算强度
        gray_im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 使用人脸检测器检查人脸
        rects = detector(gray_im, 1)
        if len(rects) == 0:  # 没有人脸
            if int((rotate_num * rotate) / 360) > 0:  # 旋转至少360度后还没有人脸就退出
                break
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # cv2.img 转 pil.image
            img_rotate = img_pil.rotate(rotate)  # 旋转后的图片
            rotate_num += 1
            img = cv2.cvtColor(np.asarray(img_rotate), cv2.COLOR_RGB2BGR)  # pil.image转cv2.img
        else:
            image = img
            break
    return rects, image


# 测试方法
def facedetect(img_path, save_path):
    """
    人脸检测
    :param img_path:图片路径
    :param save_path:保存路径
    :return:
    """
    frame = cv2.imread(img_path)

    rects, image = face_detector(frame)
    for i, d in enumerate(rects):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0

        face = image[x1:y1, x2:y2]
        # 调整图片的对比度与亮度， 对比度与亮度值都取随机数
        face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
        face = cv2.resize(face, (size, size))
        cv2.imwrite(save_path, face)


def catchpicfromvideo1(video_path, file_path):
    """
    由于视频方向是反的，每帧图需要旋转，很麻烦且耗时，现采取每帧图形保存，手动批量旋转保存图片
    :param video_path: video路径
    :param file_path: 人脸保存路径
    :return:
    """

    # 视频来源
    cap = cv2.VideoCapture(video_path)

    num = 331

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        img_name = '%s/%d.jpg' % (file_path, num)
        num += 1
        cv2.imwrite(img_name, frame)
        print(num)

    cap.release()
    cv2.destroyAllWindows()


def catch_face(frame_path, save_path):
    """
    人脸截取
    :param frame_path:每帧图片保存路径
    :param save_path: 人脸截图保存路径
    :return:
    """
    list = os.listdir(frame_path)
    for i in range(0, len(list)):
        path = os.path.join(frame_path, list[i])
        facedetect(path, os.path.join(save_path, '%d.jpg' % int(round(time.time() * 1000))))
        print(i)


class myThread1(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        catch_face('F:/AI/videos/zjq', 'F:/AI/videos/face/zjq')


class myThread2(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        catch_face('F:/AI/videos/zgc', 'F:/AI/videos/face/zgc')


if __name__ == '__main__':
    # catchpicfromvideo("F:/AI/videos/zgc.mp4", "F:/AI/videos/zgc")
    # catchpicfromvideo1("F:/AI/videos/zjq1.mp4", "F:/AI/videos/zjq")
    # catch_face('F:/AI/videos/zjq', 'F:/AI/videos/face/zjq')
    # facedetect('F:/AI/videos/zgc/0.jpg', 'F:/temp/zgc.jpg')
    # rects = face_detector(cv2.imread('F:/AI/videos/zgc/0.jpg'))
    # print(len(rects))
    # 创建新线程
    thread1 = myThread1(1, "Thread-1", 1)
    thread2 = myThread2(2, "Thread-2", 2)

    # 开启线程
    thread1.start()
    thread2.start()
