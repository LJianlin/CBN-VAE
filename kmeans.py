# -*- coding: utf-8 -*-
"""

    2019/04/17
    Liu Jianlin
"""
import numpy as np
import time
import matplotlib.pyplot as plt


# calculate Euclidean distance  
def eucldistance(vector1, vector2):
	return abs(vector2 - vector1)  # 求这两个矩阵的距离，vector1、2均为矩阵


# init centroids with random samples  
# 在样本集中随机选取k个样本点作为初始质心
def initcentroids(dataSet, k):
	numSamples, dim = dataSet.shape  # 矩阵的行数、列数
	centroids = np.zeros((k, dim))  # 感觉要不要你都可以
	for i in range(k):
		index = int(np.random.uniform(0, numSamples))  # 随机产生一个浮点数，然后将其转化为int型
		centroids[i, :] = dataSet[index, :]
	return centroids

####
# k-means cluster
# dataSet为一个矩阵
# k为将dataSet矩阵中的样本分成k个类
def kmeans(dataSet, k):
	numSamples = dataSet.shape[0]  # 读取矩阵dataSet的第一维度的长度,即获得有多少个样本数据
	#  first column stores which cluster this sample belongs to,
	#  second column stores the error between this sample and its centroid
	clusterAssment = np.mat(np.zeros((numSamples, 2)))  # 得到一个N*2的零矩阵
	clusterChanged = True

	## step 1: init centroids
	centroids = initcentroids(dataSet, k)  # 在样本集中随机选取k个样本点作为初始质心
	# print('centroids')
	# print(centroids)
	while clusterChanged:

		clusterChanged = False
		## for each sample
		for i in range(numSamples):  # range
			minDist = 100000.0
			minIndex = 0
			## for each centroid
			# ## step 2: find the centroid who is closest
			# #计算每个样本点与质点之间的距离，将其归内到距离最小的那一簇
			for j in range(k):
				distance = eucldistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist = distance
					minIndex = j
			## step 3: update its cluster
			# k个簇里面与第i个样本距离最小的的标号和距离保存在clusterAssment中
			# 若所有的样本不在变化，则退出while循环
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist  # 两个**表示的是minDist的平方

		## step 4: update centroids
		for j in range(k):
			# clusterAssment[:,0].A==j是找出矩阵clusterAssment中第一列元素中等于j的行的下标，返回的是一个以array的列表，第一个array为等于j的下标
			pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 将dataSet矩阵中相对应的样本提取出来
			centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 计算标注为j的所有样本的平均值

	print('Congratulations, cluster complete!')
	return centroids, clusterAssment


def kmeans_cel(x, cel_num, k, save_path):
	print('x.shape:' + str(x.shape))
	sample_num = int(x.shape[0]*x.shape[1])
	fc1_w = x.reshape([sample_num])
	print('x.reshape.shape:' + str(fc1_w.shape))
	print('cel_num:' + str(cel_num))
	print('k:' + str(k))
	index_array = np.argsort(fc1_w)
	fc1_w.sort()
	num = int(np.ceil((sample_num)/cel_num))
	a = np.zeros([num, 1])
	for i in range(0, num-1):
		a[i] = np.mean(fc1_w[i*cel_num:(i+1)*cel_num])
	a[-1] = np.mean(fc1_w[(num-1)*cel_num:])

	centroids, clusterAssment = kmeans(a, k)  #调用KMeans文件中定义的kmeans方法。
	clusterAssment = clusterAssment[:, 0]

	mask = np.zeros([sample_num, 1])
	for i in range(0, num-1):
		for j in range(0, cel_num):
			mask[index_array[i*cel_num+j]] = clusterAssment[i]
	mask[(num-1)*cel_num:] = clusterAssment[-1]
	print('mask.shape:' + str(mask.shape))
	np.savez(save_path, fc1_w_clusterAssment=mask)


# show your cluster only available with 2-D data
# centroids为k个类别，其中保存着每个类别的质心
# clusterAssment为样本的标记，第一列为此样本的类别号，第二列为到此类别质心的距离
def showCluster(dataSet, k, centroids, clusterAssment):
	numSamples, dim = dataSet.shape

	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	if k > len(mark):
		print("Sorry!")
		return 1

		# draw all samples
	for i in range(numSamples):
		markIndex = int(clusterAssment[i, 0])  # 为样本指定颜色
		plt.plot(dataSet[i], mark[markIndex])

	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	# draw the centroids
	for i in range(k):
		plt.plot(centroids[i], mark[i], markersize=12)

	plt.show()

