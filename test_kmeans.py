# -*- coding: utf-8 -*-
"""

    2019/04/17
    Liu Jianlin
"""
  
import numpy as np
import time  
import matplotlib.pyplot as plt 
from kmeans import *
   
## step 1: load data  
# print ("step 1: load data..." )
weight = np.load("./save_np/fc_w_noprune.npz")
fc1_w = np.mat(weight['fc3_w'])
fc1_w = fc1_w.reshape([192*10, 1])
print(fc1_w.shape)
print(type(fc1_w))
print(fc1_w)
## step 2: clustering...
print ("step 2: clustering..."  )

k = 10
centroids, clusterAssment = kmeans(fc1_w, k)  #调用KMeans文件中定义的kmeans方法。
clusterAssment = clusterAssment[:, 0]
print(centroids)
print(clusterAssment)
np.savez("./save_kmeans/fc3_w_clusterAssment.npz", fc1_w_clusterAssment=clusterAssment)
## step 3: show the result
# print ("step 3: show the result..."  )
# showCluster(fc1_w, k, centroids, clusterAssment)

# fc1_w_mask = np.load("./save_kmeans/fc2_w_clusterAssment.npz")
# fc1_w_clusterAssment = np.mat(fc1_w_mask['fc1_w_clusterAssment'])
# a = fc1_w_clusterAssment.reshape([120, 84])
# print(a.shape)
# print(type(a))
# print(a)
# k = 0
# for i in range(a.shape[0]):
# 	for j in range(a.shape[1]):
# 		if a[i, j] == 0:
# 			fc2_w[i, j] = 0.
# 			k += 1
#
# print(fc2_w)
# print(k)