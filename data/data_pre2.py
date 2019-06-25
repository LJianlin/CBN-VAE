# -*- coding: utf-8 -*-
"""
    原始数据处理
    2018/10/15
    Liu Jianlin
"""

import numpy as np
import matplotlib.pyplot as plt


def data_preprocess():
    f = open("/home/ljl/RBM_VAE/data/get.txt", 'r')
    so = f.readlines()
    f.close()
    print(len(so))
    result = []
    for line in so:
        data = list(map(float, line.split()))
        result.append(data)
    data1 = np.array(result, dtype=float)
    data1 = data1.reshape([len(so), 1])
    print("原始数据长度：%d" % len(so))

    data007_thr = []
    data007_thr_num = 0
    for j in range(0, len(so)):
        if (data1[j] >= -5) and (data1[j] <= 45):
            data007_thr_num = data007_thr_num + 1
            data007_thr.append(data1[j])
    data007_thr = np.array(data007_thr, dtype=float)
    print("阈值处理后数据长度：%d" % data007_thr_num)

    data007 = []
    data007_num = 0
    mean_data007_thr = np.mean(data007_thr)
    std_data007_thr = np.std(data007_thr)
    print(mean_data007_thr)
    print(std_data007_thr)
    for k in range(0, data007_thr_num):
        if abs(data007_thr[k] - mean_data007_thr) <= abs(3 * std_data007_thr):
            data007_num = data007_num + 1
            data007.append(data007_thr[k])
    data007 = np.array(data007, dtype=float)
    print("3倍标准差处理后数据长度：%d" % data007_num)

    data0 = []
    data0_num = 0
    data1 = []
    data1_num = 0
    for k in range(1, data007_num):
        if abs(data007[k-1] - data007[k]) <= 0.5:
            data0_num = data0_num + 1
            data0.append(data007[k])
    data0 = np.array(data0, dtype=float)
    for j in range(3, data0_num-3):
        data0[j] = (data0[j-2] + data0[j-1] + data0[j] + data0[j+1] + data0[j+2] + data0[j+3] + data0[j-3]) / 7

    for k in range(1, data0_num):
        if abs(data0[k-1] - data0[k]) <= 0.5:
            data1_num = data1_num + 1
            data1.append(data0[k])
    data0 = np.array(data0, dtype=float)
    for j in range(3, data1_num-3):
        data1[j] = (data1[j-2] + data1[j-1] + data1[j] + data1[j+1] + data1[j+2] + data1[j+3] + data1[j-3]) / 7
    print("变化率阈值处理后数据长度：%d" % data0_num)

    # with open("get.txt", "w") as f:
    #     for j in range(data1_num):
    #         f.write(str(float(data1[j])) + '\n')

    # data007_o = []
    # seqdim = int(120 * 24)
    # seqnum = int(np.floor(data0_num / seqdim))
    # train_element_num = int(seqdim * seqnum)
    # data007_t = data0.copy()
    # data007_t = data007_t[0:train_element_num, :]
    # max_train = max(data007_t)
    # min_train = min(data007_t)
    # print(max_train)
    # print(min_train)
    # for i in range(0, train_element_num):
    #     data007_o.append((data007_t[i] - min_train) / (max_train - min_train))
    # data007_o = np.array(data007_o, dtype=float)
    # print("数据集长度：%d" % train_element_num)

    # #划分训练集和测试集
    # seqnum_train = int(seqnum * 0.9)
    # data = data007_o.copy()
    # data = data.reshape([seqnum, seqdim])
    # data_train = data[0:seqnum_train, :]
    # data_test = data[seqnum_train:, :]
    # #print("训练集数据长度：%d" % seqnum_train)
    # print("训练集尺寸：" + str(data_train.shape))
    # print("测试集尺寸：" + str(data_test.shape))
	#
    # return data_train, data_test

    plt.subplots(2, 2, figsize=(12, 6))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    plt.sca(ax1)
    plt.plot(data1)
    ax1.set_title("Original temperature data")
    plt.sca(ax2)
    plt.plot(data007_thr)
    ax2.set_title("After threshold processing")
    plt.sca(ax3)
    plt.plot(data007)
    ax3.set_title("After three times std")
    plt.sca(ax4)
    plt.plot(data0)
    ax4.set_title("After max-min normalized")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()

data_preprocess()