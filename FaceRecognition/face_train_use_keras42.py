#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/2 16:59
# @Author  : Guicheng.Zhou
# @File    : face_train_use_keras42.py
# @Software: IntelliJ IDEA
# @Describe: 使用Keras训练人脸数据

import random
import cv2

from FaceRecognition.load_face_dataset41 import load_dataset, resize_image, IMAGE_SIZE
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


class DataSet:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None

        # 验证集
        self.valid_images = None
        self.valid_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        # 数据集加载路径
        self.path_name = path_name

        # 当前库采用的维度顺序
        self.input_shape = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作,交叉验证的原则将数据集划分成三部分：训练集、验证集、测试集；
    # 交叉验证属于机器学习中常用的精度测试方法，它的目的是提升模型的可靠和稳定性。
    # 我们会拿出大部分数据用于模型训练，小部分数据用于对训练后的模型验证，
    # 验证结果会与验证集真实值（即标签值）比较并计算出差平方和，此项工作重复进行，
    # 直至所有验证结果与真实值相同，交叉验证结束，模型交付使用
    # 类别数量 nb_classes
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        # 加载数据到内存
        images, labels = load_dataset(self.path_name)

        # 第一步：交叉验证
        # test_size参数按比例划分数据集,20%的数据用于验证，80%用于训练模型
        # 参数random_state用于指定一个随机数种子，从全部数据中随机选取数据建立训练集和验证集
        # train_test_split()函数会按照训练集特征数据（这里就是图像数据）、测试集特征数据、训练集标签、测试集标签的顺序返回各数据集
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        # 第二步:按照keras库运行的后端系统要求改变图像数据的维度顺序
        # keras建立在tensorflow或theano基础上,，换句话说，keras的后端系统可以是tensorflow也可以是theano,‘th’代表theano，'tf'代表tensorflow
        # 后端系统决定了图像数据输入CNN网络时的维度顺序，当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        # 通过numpy提供的reshape()函数重新调整数组维度
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

        # 输出训练集、验证集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')

        # 第三步：将数据标签进行one-hot编码，使其向量化
        # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)

        # 第四步：归一化图像数据
        # 像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        # 将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels


# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=2):
        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()

        # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        # 网络模型共18层，包括4个卷积层、5个激活函数层、2个池化层（pooling layer）、3个Dropout层、2个全连接层、1个Flatten层、1个分类层
        # 卷积层包含32个卷积核，每个卷积核大小为3x3，border_mode值为“same”意味着我们采用保留边界特征的方式滑窗，而值“valid”则指定丢掉边界像素
        self.model.add(Conv2D(32, (3, 3), input_shape=dataset.input_shape, padding='same'))  # 1 2维卷积层
        self.model.add(Activation(
            'relu'))  # 2 激活函数层,采用的relu（Rectified Linear Units，修正线性单元，ƒ(x) = max(0, x)）函数,小于0的输入，输出全部为0，大于0的则输入与输出相等，该函数的优点是收敛速度快。

        self.model.add(Conv2D(32, (3, 3)))  # 3 2维卷积层
        self.model.add(Activation('relu'))  # 4 激活函数层

        # 池化层存在的目的是缩小输入的特征图，简化网络计算复杂度；同时进行特征压缩，突出主要特征。
        # 我们通过调用MaxPooling2D()函数建立了池化层，这个函数采用了最大值池化法，这个方法选取覆盖区域的最大值作为区域主要特征组成新的缩小后的特征图
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 5 池化层  64*64 池化后变成 32*32
        self.model.add(Dropout(0.25))  # 6 Dropout层，流失层 随机断开一定百分比的输入神经元链接，以防止过拟合，导致过拟合这种现象的原因是模型的参数很多，但训练样本太少，导致模型拟合过度

        self.model.add(Conv2D(64, (3, 3), padding='same'))  # 7 2维卷积层
        self.model.add(Activation('relu'))  # 8 激活函数层

        self.model.add(Conv2D(64, (3, 3)))  # 9 2维卷积层
        self.model.add(Activation('relu'))  # 10 激活函数层

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 11 池化层
        self.model.add(Dropout(0.25))  # 12 Dropout层

        self.model.add(Flatten())  # 13 Flatten层 全连接层要求输入的数据必须是一维的，因此，我们必须把输入数据“压扁”成一维后才能进入全连接层，Flatten层的作用即在于此
        self.model.add(Dense(512))  # 14 Dense层，又被称作全连接层 全连接层的作用就是用于分类或回归，对于我们来说就是分类。这个函数的一个必填参数就是神经元个数，其实就是指定该层有多少个输出
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.5))  # 16 Dropout层
        self.model.add(Dense(nb_classes))  # 17 Dense层
        self.model.add(Activation('softmax'))  # 18 分类层，输出最终结果

        # 输出模型情况
        self.model.summary()

    # 参数batch_size指定每次迭代训练样本的数量
    # nb_epoch指定模型需要训练多少轮次
    def train(self, dataset, batch_size=20, nb_epoch=25, data_augmentation=True):
        # 随机梯度下降法 lr用于指定学习效率 decay指定每次更新后学习效率的衰减值，这个值一定很小（1e-6，0.000 001），否则速率会衰减很快
        # momentum指定动量值，用它来模拟物体运动时的惯性，让优化器在一定程度上保留之前的优化方向，同时利用当前样本微调最终的优化方向，这样即能增加稳定性，提高学习速度，又在一定程度上避免了陷入局部最优陷阱。
        # 参数momentum即用于指定在多大程度上保留原有方向，其值为0~1之间的浮点数。一般来说，选择一个在0.5~0.9之间的数即可。
        # 参数nesterov则用于指定是否采用nesterov动量方法，nesterov momentum是对传统动量法的一个改进方法，其效率更高
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        # loss，它用于指定一个损失函数。所谓损失函数，通俗地说，它是统计学中衡量损失和错误程度的函数，显然，其值越小，模型就越好。
        # 参数metrics用于指定模型评价指标，参数值”accuracy“表示用准确率来评价
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # 完成实际的模型配置工作

        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的训练数据，有意识的提升训练数据规模，增加训练数据量
        if not data_augmentation:
            # shuffle参数用于指定是否随机打乱数据集
            self.model.fit(dataset.train_images, dataset.train_labels, batch_size=batch_size, nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels), shuffle=True)
        # 使用实时数据提升
        else:
            datagen = ImageDataGenerator(featurewise_center=False,  # 是否使输入数据去中心化（均值为0）
                                         samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                                         featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                                         samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                                         zca_whitening=False,  # 是否对输入数据施以ZCA白化
                                         rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                                         width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                                         height_shift_range=0.2,  # 同上，只不过这里是垂直
                                         horizontal_flip=True,  # 是否进行随机水平翻转
                                         vertical_flip=False)  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels, batch_size=batch_size),
                                     epochs=nb_epoch,
                                     steps_per_epoch=int(dataset.train_images.shape[0] / batch_size),
                                     validation_data=(dataset.valid_images, dataset.valid_labels))

    MODEL_PATH = 'data/face.model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # 识别人脸
    def face_predict(self, image):
        # 依然是根据后台系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像(0,1,2,3)概率各为多少
        result = self.model.predict_proba(image)
        print('result:', result)

        # 给出类别预测：0,1,2,3
        predictResult = self.model.predict_classes(image)

        # 返回类别预测结果
        return predictResult[0], result[0][predictResult[0]]


if __name__ == '__main__':
    dataset = DataSet('F:/AI/videos/face')
    dataset.load()

    # 训练模型并保存
    model = Model()
    model.build_model(dataset)

    model.train(dataset)
    model.save_model()
    '''

    # 评估模型
    model = Model()
    model.load_model()
    model.evaluate(dataset)
    '''

    '''
    model = Model()
    model.load_model()
    predictResult, probability = model.face_predict(cv2.imread('F:/AI/traindata/zgc_test.png'))
    print(predictResult, probability)
    '''
