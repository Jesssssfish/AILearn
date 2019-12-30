#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/2 15:22
# @Author  : Guicheng.Zhou
# @File    : 3.CatchPICFromVideoUseOpenCV.py
# @Software: IntelliJ IDEA
# @Describe: 通过视频截取(使用opencv)图片并保存，用于数据训练

import cv2
import os
import time
import threading

size = 64


def catchpicfromvideo(window_name, camera_index, catch_pic_num, path_name):
    """
    :param window_name: 窗口名字
    :param camera_index: 摄像头索引号，一般是0，如果0不行可以试试1、2
    :param catch_pic_num: 需要截取的图片数量
    :param path_name: 保存路径
    :return:
    """
    cv2.namedWindow(window_name)

    # 视频来源,可以来自一段已存好的视频，也可以直接使用摄像头
    cap = cv2.VideoCapture(camera_index)

    # 告诉OpenCV使用人脸识别分类器
    classifier = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画的边框颜色，RGB格式
    color = (0, 255, 0)

    num = 0

    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break
        # 将当前帧转换成灰度图像,减少计算强度
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 人脸检测，1.2和3分别表示图像缩放比例和需要检测的有效点数
        faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0表示检测数人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect

                # 将当前帧保存为图片
                img_name = '%s/%d.jpg' % (path_name, num)
                image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
                cv2.imwrite(img_name, image)

                num += 1
                if num > catch_pic_num:  # 如果超出指定最大保存数量退出循环
                    break

                # 画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)  # 2表示矩形线条粗细
                # 显示当前捕捉到多少人脸图片了
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % num, (x + 30, y + 30), font, 1, (255, 0, 255), 4)

        if num > catch_pic_num:
            break

        # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


def facedetect(img_path, save_path):
    """
   人脸检测
   :param img_path:图片路径
   :param save_path:保存路径
   :return:
   """
    frame = cv2.imread(img_path)
    # 告诉OpenCV使用人脸识别分类器
    classifier = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")
    # 将当前帧转换成灰度图像,减少计算强度
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 人脸检测，1.2和3分别表示图像缩放比例和需要检测的有效点数
    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects) > 0:  # 大于0表示检测数人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect

            # 将当前帧保存为图片
            image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            res = cv2.resize(image, (size, size))
            cv2.imwrite(save_path, res)


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
    # catchpicfromvideo('CatchPIC', 0, 1200, 'F:/FaceData/zjq')
    # 创建新线程
    thread1 = myThread1(1, "Thread-1", 1)
    thread2 = myThread2(2, "Thread-2", 2)

    # 开启线程
    thread1.start()
    thread2.start()
