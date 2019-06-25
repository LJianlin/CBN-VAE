# -*- coding: utf-8 -*-
"""

    2018/12/25
    Liu Jianlin
"""

from data import *
from cbn_vae import *
import tensorflow as tf
import numpy as np
import time
import random
import matplotlib.pyplot as plt

#
train_set, test_set, max_min = data_read_7()
train_set1, test_set1, max_min1 = data_preprocess_all(1)

pro = 0.05 * 8
number_sam = int(2400 * pro)
print(number_sam)
mamin = max_min[0] - max_min[1]

data_7 = np.reshape(test_set, 71 * 120)
for i in range(0, 71 * 120):
	data_7[i] = data_7[i] * mamin + max_min[1]

mamin1 = max_min1[0] - max_min1[1]
data_1 = np.reshape(test_set1, 58 * 120)
for i in range(0, 58 * 120):
	data_1[i] = data_1[i] * mamin1 + max_min1[1]

index = random.sample(range(0, 2400), number_sam)
print(len(index))
print(index)



def noise3():
	b = data_7[2000:4400]
	b1 = data_1[0:2400]
	c = b.copy()
	for i in range(2400):
		if i in index:
			b[i] = b1[i]

	plt.plot(b)
	plt.plot(c)
	plt.show()

	for i in range(2400):
		print(b[i])


def noise2():
	b = data_7[2000:4400]
	c = b.copy()
	for i in range(2400):
		if i in index:
			rand_data1 = np.random.uniform(-1, 1, size=1)
			if rand_data1 >=0:
				b[i] = b[i] + 0.25 * b[i]
			else:
				b[i] = b[i] - 0.25 * b[i]


	plt.plot(b)
	plt.plot(c)
	plt.show()

	for i in range(2400):
		print(b[i])


def noise1():
	num = 0
	b = data_7[2000:4400]
	c = b.copy()
	for i in range(2400):
		if i in index:
			rand_data = np.random.normal(loc=21.7135961424, scale=6.558, size=1)
			rand_data1 = np.random.uniform(21.7135961424 - 6.558,  21.7135961424 + 6.558, size=1)
			rand_data = 0.5*rand_data + 0.5*rand_data1
			if rand_data >= 21.7135961424 + 6.558:
				rand_data = 21.7135961424 + 6.558
			if rand_data <= 21.7135961424 - 6.558:
				rand_data = 21.7135961424 - 6.558
			b[i] = rand_data
			num += 1
	print(num)
	plt.plot(b)
	plt.plot(c)
	plt.show()

	for i in range(2400):
		print(b[i])


noise2()

# count, bins, ignored = plt.hist(rand_data, 10, normed=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
# plt.show()


# print(a)
# print(a.shape)
# for i in range(0, a.shape[0]):
# 	# a[i] = a[i] * max_min + max_min[1]
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
# # if __name__ == '__main__':
#
#     logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#
#     data = input_data.read_data_sets("MNIST_data/", one_hot=True)
#     img = data.train.images[0].reshape(28,28) # a image
#     img = img -0.5
#
#     # test it..
#     filter_shape = (4, 4)
#     visible_shape = (28, 28)
#     k = 6
#     params_id = 'test1'
#
#     # testing runnability of crbm
#     crbm = CRBM(filter_shape, visible_shape, k, params_id)
#     hidden = crbm.generate_hidden_units(np.expand_dims(img,0))
#     rec = crbm.generate_visible_units(hidden, 1)
#     print('CRBM running ok')
#
#     #### test Trainer
#     crbm = CRBM(filter_shape, visible_shape, k, params_id)
#     trainer = crbm.Trainer(crbm)
#     img_fitted = np.expand_dims(img,0)
#     for _ in range(40):
#         trainer.train(img_fitted,sigma=0.01, lr=0.000001)
#
#     if plot_enabled:
#         hidden = crbm.generate_hidden_units(img_fitted)
#         rec = crbm.generate_visible_units(hidden, 1)
#         plt.figure()
#         plt.suptitle('original image')
#         plt.imshow(img)
#         plt.figure()
#         plt.suptitle('reconstructed image')
#         plt.imshow(rec[0,:,:])
#         plt.show()
#
#     #### test Saver
#     Saver = CRBM.Saver
#     save_path = 'save/w'
#     Saver.save(crbm, save_path, 0)
#
#     crbm = CRBM(filter_shape, visible_shape, k, params_id)
#     Saver.restore(crbm, save_path+'-0')
#     # resume training with
#     for _ in range(40):
#         trainer.train(img_fitted,sigma=0.01, lr=0.000001)
#
#     #### test Trainer with summary.
#     summary_dir = 'test/summaries/params_1/'
#     trainer = crbm.Trainer(crbm, summary_enabled=True, summary_dir=summary_dir)
#     for _ in range(10):
#         trainer.train(img_fitted,sigma=0.01, lr=0.000001)
#         print('see {0} with tensorboard'.format(summary_dir))
##################################################################
#
# print("请输入数据，以空格作为数据间隔。")
# data = [int(n) for n in input().split()]
#
# error_flag = False
# data_num = len(data)
# dif_num = len(set(data))
# left_index = 0
# right_index = data_num
#
# for i in range(data_num):
# 	if(data[i]>2147483647) or (data[i]<-2147483648):
# 		print("输入数据有误！")
# 		error_flag = True
# 		break
# if (error_flag == False):
# 	for k in range(data_num, 0, -1):
# 		if len(set(data[0:k])) == dif_num:
# 			right_index = k
#
# 	for j in range(0, right_index, 1):
# 		if len(set(data[j:right_index])) == dif_num:
# 			left_index = j
#
# 	len_min = right_index - left_index
# 	print("最短子序列长度为：%d" % len_min)
#
# 	for j in range(0, data_num, 1):
# 		if (len(set(data[j:j+len_min])) == dif_num):
# 			print("最短子序列为：" + str(data[j:j+len_min]) + ",起始下标为：" + str(j))
#
#
#
#
# # for j in range(0, len(a)-len(set1), 1):
# 	if len(set(a[j:j+3])) == len(set1):
# 		result.append(a[j:j+3])

	#if (a[j] in set1) and (not (a[j] in set(result1))):
	#result1.append(a[j:i-1])
	# i = len(a)-j
	# print(a[j:i-1])
	# i -= 1

# # for j in range(0, num_qe, 1):
# #
# # 	if j-num+1 < len(a):
# # 		result1.append(a[j-num+1])
# # 		print(result1)
# # 		#if (a[j] in set1) and \
#
# while len(set(result1)) <= len(set1):
# 	for j in range(0, len(a), 1):
# 		result1.append(a[j - num + 1])
# 		result.append(result1)
# 		#print(result1)
# 		num = len(result1)
# 		result1 = []
# 			#j=1
# 			#continue
			#result1.append(a[j])

# for j in range(len(a), 0, -1):
# 	if i < (j-len(set1)+1) and len(set(a[i:j])) == len(set1):
# 		result.append(a[i:j])

# if(error_flag == 0):
# 	leng = len(min(result, key=len))
# 	print("最短子序列长度为：%d"%leng)
# 	for k in result:
# 		if len(k) == leng:
# 			print("最短子序列为："+str(k))
