#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 14:33
# @Author  : Guicheng.Zhou
# @File    : face_train_use_keras43.py
# @Software: IntelliJ IDEA
# @Describe: 人脸识别

from FaceRecognition.face_train_use_keras42 import Model
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

if __name__ == '__main__':

    # 加载模型
    model = Model()
    model.load_model()

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    lt95 = 0  # 小于95概率次数
    total = 0  # 总识别次数
    # 循环检测识别人脸
    while True:
        _, frame = cap.read()
        # 图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用人脸识别分类器，读入分类器
        classifier = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")
        # 利用分类器识别出哪个区域为人脸
        faceRects = classifier.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            print("总识别次数：%d,小于95概率次数：%d" % (total, lt95))
            for faceRect in faceRects:
                total += 1
                x, y, w, h = faceRect
                # 截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID, probability = model.face_predict(image)  # 分析结果、概率
                probability = round(probability * 100, 4)
                if probability < 95:  # 概率小于95%不显示结果
                    lt95 += 1
                    continue
                name = 'other'
                if faceID == 0:
                    name = '周贵成\n%s%%' % probability
                elif faceID == 1:
                    name = '朱加强\n%s%%' % probability

                cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                pil_im = Image.fromarray(cv2_im)
                draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
                font = ImageFont.truetype('data/FZSTK.TTF', 20, encoding='utf-8')  # 第一个参数为字体文件路径，第二个为字体大小
                draw.text((x + 30, y + 30), name, (0, 0, 0), font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字
                frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)

        cv2.imshow("FacePredict", frame)
        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(5)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
