#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/2 14:44
# @Author  : Guicheng.Zhou
# @File    : 2.CatchVideoAndMark.py
# @Software: IntelliJ IDEA
# @Describe: 捕捉视频流并标记

import cv2


def catchvideandmark(window_name, camera_index):
    """
        :param window_name:窗口名字
        :param camera_index: 摄像头索引号，一般是0，如果0不行可以试试1、2
        :return:
        """
    cv2.namedWindow(window_name)

    # 视频来源,可以来自一段已存好的视频，也可以直接使用摄像头
    cap = cv2.VideoCapture(camera_index)

    # 告诉OpenCV使用人脸识别分类器
    classifier = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画的边框颜色，RGB格式
    color = (0, 255, 0)

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
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)  # 2表示矩形线条粗细

        # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    catchvideandmark('CatchVideoAndMark', 0)
