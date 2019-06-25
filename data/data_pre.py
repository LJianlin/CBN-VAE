# -*- coding: utf-8 -*-
"""
    Data Preprocessing
    2018/12/25
    Liu Jianlin
"""

import numpy as np
import matplotlib.pyplot as plt


def data_preprocess(file_name):

    f = open(file_name, 'r')
    f_data = f.readlines()
    f.close()

    result = []
    for line in f_data:
        data = list(map(float, line.split()))
        result.append(data)
    data1 = np.array(result, dtype=float)
    data1 = data1.reshape([len(f_data), 1])
    print("原始数据长度：%d" % len(f_data))

    data007_thr = []
    data007_thr_num = 0
    for j in range(0, len(f_data)):
        if (data1[j] >= -5) and (data1[j] <= 45):
            data007_thr_num = data007_thr_num + 1
            data007_thr.append(data1[j])
    data007_thr = np.array(data007_thr, dtype=float)
    print("阈值处理后数据长度：%d" % data007_thr_num)

    data007 = []
    data007_num = 0
    mean_data007_thr = np.mean(data007_thr)
    std_data007_thr = np.std(data007_thr)
    for k in range(0, data007_thr_num):
        if abs(data007_thr[k] - mean_data007_thr) <= abs(3 * std_data007_thr):
            data007_num = data007_num + 1
            data007.append(data007_thr[k])
    data007 = np.array(data007, dtype=float)
    print("3倍标准差处理后数据长度：%d" % data007_num)

    return data007


def data_save(save_path="./data6-15.npz", dim=10, beilv=1):

    data1 = np.zeros((dim, 100000), dtype=np.float32)
    data_num = []
    for i in range(6, 16, 1):
        file_name = "/home/ljl/CRBM_AE/dataset/data" + str(i) + "/temperature.txt"
        data = data_preprocess(file_name)
        for j in range(data.shape[0]):
            data1[i - 6, j] = data[j]

        data_num.append(data.shape[0])
    print("Data length：" + str(data_num))
    print("Min data length：%d" % min(data_num))

    seqdim = int(120 * beilv)
    seqnum = int(np.floor(min(data_num) / seqdim))
    train_element_num = int(seqdim * seqnum)
    print("Data Set number :%d" % train_element_num)
    data_train = np.zeros((dim, train_element_num), dtype=np.float32)
    max_min = np.zeros((dim, 2), dtype=np.float32)
    data_t = data1.copy()

    for ii in range(dim):
        data_train[ii, :] = data_t[ii, 0:train_element_num]
        max_train = max(data_train[ii, :])
        min_train = min(data_train[ii, :])
        print("max_train:%d" % max_train)
        print("min_train:%d" % min_train)
        max_min[ii, 0] = max_train
        max_min[ii, 1] = min_train
        for jj in range(0, train_element_num):
            data_train[ii, jj] = (data_train[ii, jj] - min_train) / (max_train - min_train)

    print(data_train.shape)

    # 划分训练集和测试集
    seqnum_train = int(seqnum * 0.8)
    data_all = data_train.copy()
    data_all = data_all.reshape([seqnum, dim, seqdim])
    train_set = data_all[0:seqnum_train, :, :]
    test_set = data_all[seqnum_train:, :, :]
    print("训练集尺寸：" + str(train_set.shape))
    print("测试集尺寸：" + str(test_set.shape))

    np.savez(save_path, train_set=train_set, test_set=test_set, max_min=max_min)

    # plt.plot(train_set[0, 0, :])
    # plt.show()
    # plt.plot(data1[0, 0:120])
    # plt.show()


def data_read(file_path="./data6-15.npz"):

    data = np.load(file_path)
    train_set = data["train_set"]
    test_set = data["test_set"]
    max_min = data["max_min"]

    return train_set, test_set, max_min


def data_read_7(beilv=1):

    f = open("./dataset/data7/temperature.txt", 'r')
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
        if abs(data007[k - 1] - data007[k]) <= 0.5:
            data0_num = data0_num + 1
            data0.append(data007[k])
    data0 = np.array(data0, dtype=float)
    for j in range(3, data0_num - 3):
        data0[j] = (data0[j - 2] + data0[j - 1] + data0[j] + data0[j + 1] + data0[j + 2] + data0[j + 3] + data0[
            j - 3]) / 7

    for k in range(1, data0_num):
        if abs(data0[k - 1] - data0[k]) <= 0.5:
            data1_num = data1_num + 1
            data1.append(data0[k])
    data1 = np.array(data1, dtype=float)
    for j in range(3, data1_num - 3):
        data1[j] = (data1[j - 2] + data1[j - 1] + data1[j] + data1[j + 1] + data1[j + 2] + data1[j + 3] + data1[
            j - 3]) / 7
    print("变化率阈值处理后数据长度：%d" % data1_num)

    data007_o = []
    seqdim = int(120 * beilv)
    seqnum = int(np.floor(data1_num / seqdim))
    train_element_num = int(seqdim * seqnum)
    data007_t = data1.copy()
    data007_t = data007_t[0:train_element_num, :]
    max_min = np.zeros(2, dtype=np.float32)
    max_min[0] = max(data007_t)
    max_min[1] = min(data007_t)
    print(max_min)
    for i in range(0, train_element_num):
        data007_o.append((data007_t[i] - max_min[1]) / (max_min[0] - max_min[1]))
    data007_o = np.array(data007_o, dtype=float)
    print("数据集长度：%d" % train_element_num)

    # data007_o = []
    # seqdim = int(120 * beilv)
    # seqnum = int(np.floor(data007_num / seqdim))
    # train_element_num = int(seqdim * seqnum)
    # data007_t = data007.copy()
    # data007_t = data007_t[0:train_element_num, :]
    # max_min = np.zeros(2, dtype=np.float32)
    # max_min[0] = max(data007_t)
    # max_min[1] = min(data007_t)
    # print(max_min)
    # for i in range(0, train_element_num):
    #     data007_o.append((data007_t[i] - max_min[1]) / (max_min[0] - max_min[1]))
    # data007_o = np.array(data007_o, dtype=float)
    # print("数据集长度：%d" % train_element_num)
    #print(data007_train[0:120])

    # 划分训练集和测试集
    seqnum_train = int(seqnum * 0.8)
    data = data007_o.copy()
    data = data.reshape([seqnum, 1, seqdim, 1])
    data_train = data[0:seqnum_train, :, :, :]
    data_test = data[seqnum_train:, :, :, :]
    #print("训练集数据长度：%d" % seqnum_train)
    print("训练集尺寸：" + str(data_train.shape))
    print("测试集尺寸：" + str(data_test.shape))

    return data_train, data_test, max_min


def data_preprocess_all(data_num, beilv=1):

    filename = "./dataset/data" + str(data_num) + "/temperature.txt"
    f = open(filename, 'r')
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
        if (data1[j] >= 5) and (data1[j] <= 45):
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
        if abs(data007[k - 1] - data007[k]) <= 0.5:
            data0_num = data0_num + 1
            data0.append(data007[k])
    data0 = np.array(data0, dtype=float)
    for j in range(3, data0_num - 3):
        data0[j] = (data0[j - 2] + data0[j - 1] + data0[j] + data0[j + 1] + data0[j + 2] + data0[j + 3] + data0[
            j - 3]) / 7

    for k in range(1, data0_num):
        if abs(data0[k - 1] - data0[k]) <= 0.5:
            data1_num = data1_num + 1
            data1.append(data0[k])
    data1 = np.array(data1, dtype=float)
    for j in range(3, data1_num - 3):
        data1[j] = (data1[j - 2] + data1[j - 1] + data1[j] + data1[j + 1] + data1[j + 2] + data1[j + 3] + data1[
            j - 3]) / 7
    print("变化率阈值处理后数据长度：%d" % data1_num)

    data007_o = []
    seqdim = int(120 * beilv)
    seqnum = int(np.floor(data1_num / seqdim))
    train_element_num = int(seqdim * seqnum)
    data007_t = data1.copy()
    data007_t = data007_t[0:train_element_num, :]
    max_min = np.zeros(2, dtype=np.float32)
    max_min[0] = max(data007_t)
    max_min[1] = min(data007_t)
    print(max_min)
    for i in range(0, train_element_num):
        data007_o.append((data007_t[i] - max_min[1]) / (max_min[0] - max_min[1]))
    data007_o = np.array(data007_o, dtype=float)
    print("数据集长度：%d" % train_element_num)

    # 划分训练集和测试集
    seqnum_train = int(seqnum * 0.8)
    data = data007_o.copy()
    data = data.reshape([seqnum, 1, seqdim, 1])
    data_train = data[0:seqnum_train, :, :, :]
    data_test = data[seqnum_train:, :, :, :]
    #print("训练集数据长度：%d" % seqnum_train)
    print("训练集尺寸：" + str(data_train.shape))
    print("测试集尺寸：" + str(data_test.shape))

    return data_train, data_test, max_min

def data_read_71(beilv=1):

    f = open("./data/zz2.txt", 'r')
    so = f.readlines()
    f.close()
    print(len(so))
    result = []
    # data1 = np.zeros([len(so), 1], dtype=float)
    for line in so:
        data = list(map(float, line.split()))
        # print(data)
        # data1[line] = data[0]
        result.append(data)
    # print(result)
    print(len(result))
    data1 = np.array(result, dtype=float)
    data1 = data1.reshape([len(so), 1])
    print("原始数据长度：%d" % len(so))

    data007_thr = []
    data007_thr_num = 0
    for j in range(0, len(so)):
        if (data1[j] >= -5) and (data1[j] <= 2450000):
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
        if abs(data007[k - 1] - data007[k]) <= 15000:
            data0_num = data0_num + 1
            data0.append(data007[k])
    data0 = np.array(data0, dtype=float)
    for j in range(3, data0_num - 3):
        data0[j] = (data0[j - 2] + data0[j - 1] + data0[j] + data0[j + 1] + data0[j + 2] + data0[j + 3] + data0[
            j - 3]) / 7

    for k in range(1, data0_num):
        if abs(data0[k - 1] - data0[k]) <= 15000:
            data1_num = data1_num + 1
            data1.append(data0[k])
    data1 = np.array(data1, dtype=float)
    for j in range(3, data1_num - 3):
        data1[j] = (data1[j - 2] + data1[j - 1] + data1[j] + data1[j + 1] + data1[j + 2] + data1[j + 3] + data1[
            j - 3]) / 7
    print("变化率阈值处理后数据长度：%d" % data1_num)

    data007_o = []
    seqdim = int(120 * beilv)
    seqnum = int(np.floor(data1_num / seqdim))
    train_element_num = int(seqdim * seqnum)
    data007_t = data1.copy()
    data007_t = data007_t[0:train_element_num, :]
    max_min = np.zeros(2, dtype=np.float32)
    max_min[0] = max(data007_t)
    max_min[1] = min(data007_t)
    print(max_min)
    for i in range(0, train_element_num):
        data007_o.append((data007_t[i] - max_min[1]) / (max_min[0] - max_min[1]))
    data007_o = np.array(data007_o, dtype=float)
    print("数据集长度：%d" % train_element_num)

    # data007_o = []
    # seqdim = int(120 * beilv)
    # seqnum = int(np.floor(data007_num / seqdim))
    # train_element_num = int(seqdim * seqnum)
    # data007_t = data007.copy()
    # data007_t = data007_t[0:train_element_num, :]
    # max_min = np.zeros(2, dtype=np.float32)
    # max_min[0] = max(data007_t)
    # max_min[1] = min(data007_t)
    # print(max_min)
    # for i in range(0, train_element_num):
    #     data007_o.append((data007_t[i] - max_min[1]) / (max_min[0] - max_min[1]))
    # data007_o = np.array(data007_o, dtype=float)
    # print("数据集长度：%d" % train_element_num)
    #print(data007_train[0:120])

    # 划分训练集和测试集
    seqnum_train = int(seqnum * 0.8)
    data = data007_o.copy()
    data = data.reshape([seqnum, 1, seqdim, 1])
    data_train = data[0:seqnum_train, :, :, :]
    data_test = data[seqnum_train:, :, :, :]
    #print("训练集数据长度：%d" % seqnum_train)
    print("训练集尺寸：" + str(data_train.shape))
    print("测试集尺寸：" + str(data_test.shape))

    return data_train, data_test, max_min