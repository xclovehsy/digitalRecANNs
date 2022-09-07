#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: 徐聪
# datetime: 2022-05-12 17:06
# software: PyCharm
import random

import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
import struct
import scipy.special as ssp

# 参数
rate = 0.1  # 学习率
epochs = 10  # 迭代的次数

# 输入层、隐藏层、输出层节点数量
input_num = 784
hide_num = 200
output_num = 10

# 生成隐藏层以及输出层的权重矩阵，正态分布，期望为0，方差为hide_num的-0.5次方
wih = np.random.normal(0.0, pow(hide_num, -0.5), (hide_num, input_num))
who = np.random.normal(0.0, pow(hide_num, -0.5), (output_num, hide_num))

# 激活函数
active_fun = lambda x: ssp.expit(x)


def net_train(data_list, label_list):
    """
    模型训练
    :param data_list: 一张图片的数据
    :param label_list: 标签
    :return:
    """
    global wih
    global who

    # 数据初始化
    data = np.array(data_list, ndmin=2).T
    label = np.array(label_list, ndmin=2).T

    # 前向传播
    hide_z = wih.dot(data)
    hide_a = active_fun(hide_z)
    final_z = who.dot(hide_a)
    final_a = active_fun(final_z)

    # 反向传播
    final_error = label - final_a
    hide_error = who.T.dot(final_error)

    # 利用公式计算梯度
    final_gradient = np.dot((final_error * final_a * (1.0 - final_a)), np.transpose(hide_a))
    hide_gradient = np.dot((hide_error * hide_a * (1.0 - hide_a)), np.transpose(data))

    # 梯度下降更新权重矩阵  因为前面计算err时，是真实值-预测值，所以这里加梯度
    who += rate * final_gradient
    wih += rate * hide_gradient


def net_query(data_list):
    """
    神经网络识别函数
    :param data_list: 需要识别的数据
    :return:
    """
    global wih
    global who

    # 数据初始化
    data = np.array(data_list, ndmin=2).T

    # 前向传播计算
    hide_z = wih.dot(data)
    hide_a = active_fun(hide_z)
    final_z = who.dot(hide_a)
    final_a = active_fun(final_z)

    return final_a


def load_mnist(path, kind='train'):
    """
    读取压缩包里的训练数据集
    :param path:数据集的路径
    :param kind:代表读取训练集
    :return:
    """

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    # 使用gzip打开文件
    with gzip.open(labels_path, 'rb') as lbpath:
        # 使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
        # 这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II', lbpath.read(8))
        # 使用np.fromstring读取剩下的数据，lbpath.read()表示读取所有的数据
        labels = np.fromstring(lbpath.read(), dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromstring(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def show_result():
    """
    展示识别结果
    :return:
    """
    images, labels = load_mnist(r"C:\Users\xc\Desktop\大二下\最优化\实验报告\实验4\训练集", "t10k")  # 读取测试文件数据
    fig, ax = plt.subplots(
        nrows=4,
        ncols=5,
        sharex=True,
        sharey=True,
    )

    plt.rcParams['font.sans-serif'] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    ax = ax.flatten()
    # 添加随机数种子
    random.seed()
    for i in range(20):
        index = random.randint(0, 1000)
        img = images[index].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        label = np.argmax(net_query(images[index]))
        ax[i].set_xlabel(f"识别结果：{label}")

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def main():
    # 读取文件数据
    images, labels = load_mnist(r"C:\Users\xc\Desktop\大二下\最优化\实验报告\实验4\训练集")
    image_num = images.shape[0]  # 图片数量

    # 图片数据进行预处理  让所有数据都处于0.01到1.00之间
    images = images / 255.0 * 0.99 + 0.01

    # 神经网络模型的训练
    print("模型训练开始")
    for i in range(epochs):

        # 遍历训练集中的数据
        for j in range(image_num):
            # 数据预处理
            data = images[j, :]
            label = np.ones(output_num) * 0.01
            label[labels[j]] = 0.99

            # 训练模型
            net_train(data, label)
        print(f"训练迭代次数：{i+1}")

    print("训练模型完成")

    # 测试模型
    images, labels = load_mnist(r"C:\Users\xc\Desktop\大二下\最优化\实验报告\实验4\训练集", "t10k")  # 读取测试文件数据
    image_num = images.shape[0]  # 图片数量
    images = images / 255.0 * 0.99 + 0.01
    score = 0  # 计算准确率

    for i in range(image_num):
        label = net_query(images[i])
        if np.argmax(label) == labels[i]:
            score += 1
    print(f"正确率：{score / image_num * 100}%")

    # 答应识别结果
    show_result()


if __name__ == '__main__':
    main()
