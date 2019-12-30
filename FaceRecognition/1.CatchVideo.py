#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/2 13:53
# @Author  : Guicheng.Zhou
# @File    : 1.CatchVideo.py
# @Software: IntelliJ IDEA
# @Describe: 利用OpenCV捕捉视频流并展示

import cv2


def catchvideo(window_name, camera_index):
    """
    :param window_name:窗口名字
    :param camera_index: 摄像头索引号，一般是0，如果0不行可以试试1、2
    :return:
    """
    cv2.namedWindow(window_name)

    # 视频来源,可以来自一段已存好的视频，也可以直接使用摄像头
    cap = cv2.VideoCapture(camera_index)

    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break
        # 显示图像并等待10毫秒按键输入，输入q表示退出
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    catchvideo('CatchVideo', 0)
