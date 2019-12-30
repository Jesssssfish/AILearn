#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/9 10:52
# @Author  : Guicheng.Zhou
# @File    : neuralnetwork.py
# @Software: IntelliJ IDEA
# @Describe: 自制神经网络
import numpy as np
import scipy.special


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        初始化神经网络
        :param inputnodes:输入层节点数
        :param hiddennodes: 隐藏层节点数
        :param outputnodes: 输出层节点数
        :param learningrate: 学习效率
        """
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 设置权重
        # self.wih = (np.random.rand(self.inodes, self.hnodes) - 0.5)
        # self.who = (np.random.rand(self.hnodes, self.onodes) - 0.5)
        # 使用正态分布设置权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 定义激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        """

        :param inputs_list: 训练集
        :param targets_list:目标集
        :return:
        """
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.2
    epochs = 7

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    train_data_file = open('data/mnist_train.csv', 'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()

    for e in range(epochs):
        train_index = 0
        for record in train_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            train_index += 1
            print("Train:", e, ' epochs ', train_index, '/', len(train_data_list))

    test_data_file = open('data/mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []
    test_index = 0
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        # print(correct_label, "correct label")
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        # print(label, "network`s answer")
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
        test_index += 1
        print("Query:", test_index, '/', len(test_data_list))

    scorecard_array = np.asarray(scorecard)
    print("performance= ", scorecard_array.sum() / scorecard_array.size)
