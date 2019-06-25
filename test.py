# -*- coding: utf-8 -*-
"""
    test
    2018/12/25
    Liu Jianlin
"""

from data import *
from cbn_vae import *
from kmeans import *
from retrain import *
import tensorflow as tf
import numpy as np
import time
import random
import matplotlib.pyplot as plt

ckpt_save_path = "./ckpt/save/cnn_ae_test_500"
prune_save_path = './prune_save/cnn_vae_prune_500.ckpt'
retrain_save_path = './prune_save/cnn_vae_retrain_500.ckpt'

train_set, test_set, max_min = data_read_7()
mamin = max_min[0] - max_min[1]

weight_prune(prune_file=ckpt_save_path, prune_save_path=prune_save_path)
evalu(test_set, save_path=prune_save_path)
evalu(test_set, save_path=ckpt_save_path)
for i in range(10):
	retrain(train_set, test_set, max_min, maxepoch=50, ckpt_path=prune_save_path, save_path=retrain_save_path)
	weight_prune(prune_file=retrain_save_path, prune_save_path=prune_save_path)
	evalu(test_set, save_path=prune_save_path)

# weight = np.load("./save_np/fc_w_noprune.npz")
# fc1_w = np.mat(weight['wt_fc1'])
# fc1_w = fc1_w.reshape([54*27, 1])
# print(fc1_w.shape)
# print(type(fc1_w))
# print(fc1_w)
# ## step 2: clustering...
# print ("step 2: clustering..."  )
#
# k = 10
# centroids, clusterAssment = kmeans(fc1_w, k)  #调用KMeans文件中定义的kmeans方法。
# clusterAssment = clusterAssment[:, 0]
# print(centroids)
# print(clusterAssment)
# np.savez("./save_kmeans/wt_fc1_clusterAssment.npz", fc1_w_clusterAssment=clusterAssment)


# reader = pywrap_tensorflow.NewCheckpointReader(prune_save_path)
# with tf.Session() as sess:
# 	all_variables = reader.get_variable_to_shape_map()
# 	print(all_variables)
# 	w_fc1 = reader.get_tensor("w_conv1")
# 	w5 = reader.get_tensor("w5")
# 	w6 = reader.get_tensor("w6")
# 	w7 = reader.get_tensor("w7")
# 	wt_fc1 = reader.get_tensor("wt_fc1")
# 	print(w_fc1[:, :, :, 0])
# 	np.savez("./save_np/fc_w_noprune.npz", w_fc1=w_fc1, w5=w5, w6=w6, w7=w7, wt_fc1=wt_fc1)

# train_set, test_set, max_min = data_read_71()
# train_set, test_set, max_min = data_read_71()
# # mamin = max_min[0] - max_min[1]
#
# train(train_set, test_set, max_min)

# a = [882,1212,1791,1897,2033,2134]
# b = [120,240,480,720,960,1200]
# c = [120,240,480,720,960,1200]
# for i in range(6):
# 	p = b[i]/a[i]
# 	r = b[i]/c[i]
# 	f = (2*p*r)/(p+r)
# 	print('p %6.6f r %6.6f f %6.6f'%(p,r,f))


#
# #
# train_set, test_set, max_min = data_read_7()
# mamin = max_min[0] - max_min[1]
#
# filename = "./datanoise3/datanoise3_5.txt"
# f = open(filename, 'r')
# so = f.readlines()
# f.close()
# print(len(so))
#
# result = []
# for line in so:
# 	data = list(map(float, line.split()))
# 	result.append(data)
# data1 = np.array(result, dtype=float)
# print("原始数据长度：%d" % len(so))
#
# index1 = [2058, 1042, 747, 760, 1424, 1884, 1877, 1293, 1544, 1734, 1415, 1724, 1195, 763, 2338, 1102, 154, 2353, 1891, 1614, 1429, 1776, 1137, 1822, 976, 1611, 312, 2084, 1336, 758, 1182, 206, 1726, 1607, 2300, 335, 367, 1405, 1301, 1989, 395, 568, 774, 1992, 2239, 1760, 232, 701, 149, 1344, 2244, 985, 1246, 697, 1163, 2107, 175, 725, 2355, 2306, 2159, 652, 974, 1599, 242, 1265, 912, 2220, 176, 2319, 1643, 1628, 1873, 2008, 142, 2336, 1221, 1372, 1011, 1076, 2224, 587, 1106, 531, 2276, 1774, 1046, 856, 1639, 2288, 128, 2146, 3, 1218, 762, 368, 575, 2134, 1622, 715, 1077, 1010, 1727, 452, 1640, 1811, 1772, 1119, 687, 2327, 105, 297, 1467, 667, 1603, 931, 2017, 1954, 41, 2296]
#
#
#
#
#
# test_set = test_set.reshape(8520)
#
# for i in range(0, 2400):
# 	data1[i] = (data1[i] - max_min[1]) / mamin
#
# data = data1.reshape([20, 1, 120, 1])
#
#
# dataout = evalu(data)
# print(dataout.shape)
# print(dataout)
# dataout = dataout.reshape(20*120)
# error = []
# for i in range(0, 20*120):
# 	dataout[i] = dataout[i] * mamin + max_min[1]
# 	data1[i] = data1[i] * mamin + max_min[1]
# 	test_set[i+2000] = test_set[i+2000] * mamin + max_min[1]
# 	error.append(abs(dataout[i]-data1[i]))
# error = np.array(error, dtype=float)
# print(error.shape)
# print(error)
# mean_data007_thr = np.mean(error)
# std_data007_thr = np.std(error)
# print(mean_data007_thr)
# print(std_data007_thr)
# num1 = 0
# index = []
# for i in range(0, 20*120, 1):
# 	if error[i] < (0.0397 - (3 * 0.0526)) or error[i] > (0.0397 + (3 * 0.0526)):
# 		num1 += 1
# 		index.append(i)
# print('all fault num:')
# print(num1)
# print(num1-58)
# # print(index)
#
#
# print(len(index1))
#
# num2 = 0
# for o in index1:
# 	if o in index:
# 		num2 += 1
# print('true fault num:')
# print(num2)
#
# plt.plot(data1)
# plt.plot(dataout)
# plt.plot(error)
# plt.plot(test_set[2000:4400])
# plt.show()
#
# for i in range(2400):
# 	print(dataout[i])
#
# datain = test_set.reshape(71*120)
# print(datain.shape)
#
# # print(a)
# print(dataout.shape)
# error = []
# for i in range(0, 71*120):
# 	dataout[i] = dataout[i] * mamin + max_min[1]
# 	datain[i] = datain[i] * mamin + max_min[1]
# 	error.append(abs(dataout[i]-datain[i]))
#
# error = np.array(error, dtype=float)
# mean_data007_thr = np.mean(error[2000:4400])
# std_data007_thr = np.std(error[2000:4400])
# print(mean_data007_thr)
# print(std_data007_thr)
# num = 0
# for i in range(2000, 4400, 1):
# 	if error[i]<0.0397-3*0.0526 or error[i]>0.04036+3*0.0526:
# 		num += 1
# print(num)
# num1 = 0
# for i in range(0, 71*120, 1):
# 	if error[i]<0.07671-3*0.13769 or error[i]>0.07671+3*0.13769:
# 		num1 += 1
# print(num1)


# plt.plot(datain)
# plt.plot(dataout)
# plt.show()


# 	print(a[i])


# test1(train_set)
# out = test1(train_set)
# train(train_set, test_set, max_min)
# # datain = train_set.reshape([train_set.shape[0]*train_set.shape[2], 1])
# # dataout = out.reshape([train_set.shape[0]*train_set.shape[2], 1])
#
# plt.plot(datain)
# plt.plot(dataout)
# plt.show()
