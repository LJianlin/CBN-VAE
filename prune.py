# -*- coding: utf-8 -*-
"""
	Prune
	2019/04/17
	Liu Jianlin
"""
import numpy as np
from data import *
from retrain import *
from cbn_vae import *
from tensorflow.python import pywrap_tensorflow


def prune_test(index, test_set, ckpt_path):
	tf.reset_default_graph()
	reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
	conv1_wa = reader.get_tensor("w_conv1")
	# conv1_wa[:, :, :, index] = 0
	w_conv1 = tf.Variable(tf.convert_to_tensor(conv1_wa))
	b_conv1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv1")))

	conv2_wa = reader.get_tensor("w_conv2")
	# conv2_wa[:, :, :, index] = 0
	w_conv2 = tf.Variable(tf.convert_to_tensor(conv2_wa))
	b_conv2 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv2")))

	conv3_wa = reader.get_tensor("w_conv3")
	# conv3_wa[:, :, :, index] = 0
	w_conv3 = tf.Variable(tf.convert_to_tensor(conv3_wa))
	b_conv3 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv3")))

	conv4_wa = reader.get_tensor("w_conv4")
	# conv4_wa[:, :, :, index] = 0
	w_conv4 = tf.Variable(tf.convert_to_tensor(conv4_wa))
	b_conv4 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv4")))

	fc1_wa = reader.get_tensor("w_fc1")
	# fc1_w_mask = np.load("./save_kmeans/w_fc1_clusterAssment.npz")
	# fc1_w_clusterAssment = np.mat(fc1_w_mask['fc1_w_clusterAssment'])
	# fc1_w_m = fc1_w_clusterAssment.reshape([54, 27])
	# fc1_prune_num = 0
	# for i in range(0, fc1_w_m.shape[0]):
	# 	for j in range(0, fc1_w_m.shape[1]):
	# 		if (abs(fc1_w_m[i, j]) == index):
	# 			fc1_wa[i, j] = 0.
	# 			fc1_prune_num += 1
	# print(fc1_prune_num)
	w_fc1 = tf.Variable(tf.convert_to_tensor(fc1_wa))
	b_fc1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_fc1")))

	w5 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w5")))
	b5 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b5")))
	w6 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w6")))
	b6 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b6")))
	w7 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w7")))
	b7 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b7")))

	fct1_wa = reader.get_tensor("wt_fc1")
	fct1_w_mask = np.load("./save_kmeans/wt_fc1_clusterAssment.npz")
	fct1_w_clusterAssment = np.mat(fct1_w_mask['fc1_w_clusterAssment'])
	fct1_w_m = fct1_w_clusterAssment.reshape([27, 54])
	fct1_prune_num = 0
	for i in range(0, fct1_w_m.shape[0]):
		for j in range(0, fct1_w_m.shape[1]):
			if (abs(fct1_w_m[i, j]) == index):
				fct1_wa[i, j] = 0.
				fct1_prune_num += 1
	print(fct1_prune_num)
	wt_fc1 = tf.Variable(tf.convert_to_tensor(fct1_wa))

	# wt_fc1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("wt_fc1")))
	bt_fc1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("bt_fc1")))

	batch_size = 1
	x = tf.placeholder(tf.float32, shape=[batch_size, 1, 120, 1])  # train set: [289, 1, 120, 1]
	batch_num_test = int(test_set.shape[0] / batch_size)

	conv1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1, 1, 12, 1], padding='SAME') + b_conv1, name="conv1")
	o_conv1 = tf.reshape(conv1, [batch_size, 1, conv1.get_shape()[2].value * conv1.get_shape()[3].value, 1])
	o_pool1, mask1 = max_pool_1x2_with_argmax(input=o_conv1)
	conv2 = tf.nn.relu(tf.nn.conv2d(o_pool1, w_conv2, strides=[1, 1, 9, 1], padding='SAME') + b_conv2, name="conv2")
	o_conv2 = tf.reshape(conv2, [batch_size, 1, conv2.get_shape()[2].value * conv2.get_shape()[3].value, 1])
	o_pool2, mask2 = max_pool_1x2_with_argmax(input=o_conv2)
	o_pool3, mask3 = max_pool_1x2_with_argmax(input=o_pool2)
	conv3 = tf.nn.relu(tf.nn.conv2d(o_pool3, w_conv3, strides=[1, 1, 5, 1], padding='SAME') + b_conv3, name="conv3")
	o_conv3 = tf.reshape(conv3, [batch_size, 1, conv3.get_shape()[2].value * conv3.get_shape()[3].value, 1])
	o_pool4, mask4 = max_pool_1x2_with_argmax(input=o_conv3)
	conv4 = tf.nn.relu(tf.nn.conv2d(o_pool4, w_conv4, strides=[1, 1, 3, 1], padding='SAME') + b_conv4, name="conv4")
	o_conv4 = tf.reshape(conv4, [batch_size, 1, conv4.get_shape()[2].value * conv4.get_shape()[3].value, 1])
	o_pool5, mask5 = max_pool_1x2_with_argmax(input=o_conv4)
	o_pool6, mask6 = max_pool_1x2_with_argmax(input=o_pool5)
	maxpool2_flat = tf.reshape(o_pool6, [-1, o_pool6.get_shape()[2].value])
	o_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, w_fc1) + b_fc1, name="o_fc1")
	mu_encoder = tf.matmul(o_fc1, w5) + b5
	logvar_encoder = tf.matmul(o_fc1, w6) + b6
	epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')
	std_encoder = tf.exp(0.5 * logvar_encoder)
	z = mu_encoder + tf.multiply(epsilon, std_encoder)  # encoder output
	de_input = tf.nn.relu(tf.matmul(z, w7) + b7)
	ot_fc1 = tf.nn.relu(tf.matmul(de_input, wt_fc1) + bt_fc1, name="ot_fc1")
	tmaxpool2_flat = tf.reshape(ot_fc1, [batch_size, 1, o_pool6.get_shape()[2].value, 1])
	o_uppool = un_max_pool_1x2(input=tmaxpool2_flat, mask=mask6)
	o_uppool1 = un_max_pool_1x2(input=o_uppool, mask=mask5)
	r_conv4 = tf.nn.tanh(tf.reshape(o_uppool1, conv4.shape))
	ot_conv5 = tf.nn.conv2d_transpose(r_conv4, w_conv4, [batch_size, 1, o_pool4.get_shape()[2].value, 1],
									  strides=[1, 1, 3, 1], padding='SAME')
	o_uppool3 = un_max_pool_1x2(input=ot_conv5, mask=mask4)
	r_conv3 = tf.nn.tanh(tf.reshape(o_uppool3, conv3.shape))
	ot_conv4 = tf.nn.conv2d_transpose(r_conv3, w_conv3, [batch_size, 1, o_pool3.get_shape()[2].value, 1],
									  strides=[1, 1, 5, 1], padding='SAME')
	o_uppool4 = un_max_pool_1x2(input=ot_conv4, mask=mask3)
	o_uppool5 = un_max_pool_1x2(input=o_uppool4, mask=mask2)
	reshape = tf.nn.tanh(tf.reshape(o_uppool5, conv2.shape))
	ot_conv3 = tf.nn.conv2d_transpose(reshape, w_conv2, [batch_size, 1, o_pool1.get_shape()[2].value, 1],
									  strides=[1, 1, 9, 1], padding='SAME')
	o_uppool6 = un_max_pool_1x2(input=ot_conv3, mask=mask1)
	reshape = tf.nn.tanh(tf.reshape(o_uppool6, conv1.shape))
	de_output = tf.nn.conv2d_transpose(reshape, w_conv1, [batch_size, 1, 120, 1], strides=[1, 1, 12, 1], padding='SAME')

	xloss = tf.reduce_mean(abs(de_output - x))

	prd_c = tf.square(xloss)
	prd_d = tf.square(tf.reduce_mean(abs(x)))
	print('i=' + str(index))
	# outdata3 = np.zeros([batch_size, batch_num_test * 120], np.float32)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		test_errsum = 0
		prd_c_sum = 0
		prd_d_sum = 0
		for offset in range(0, batch_size * batch_num_test, batch_size):
			end = offset + batch_size
			batch_x_test = test_set[offset:end, :, :, :]
			test_xloss1, prdc, prdd = sess.run([xloss, prd_c, prd_d], feed_dict={x: batch_x_test})
			test_errsum = test_errsum + test_xloss1
			prd_c_sum = prdc + prd_c_sum
			prd_d_sum = prdd + prd_d_sum
			# outdata3[0, offset * 120:end * 120] = outdata[0, 0, :, 0]

		prd = np.sqrt(prd_c_sum / prd_d_sum) * 100
		snr = 10 * math.log((1 / (prd * prd / 10000)), 10)
		print("Test meanerror %6.6f PRD %6.6f SNR %6.6f" % (test_errsum / batch_num_test * 8.6748982, prd, snr))


if __name__ == '__main__':

	ckpt_save_path = "./ckpt/save/cnn_ae_test_500"
	prune_save_path = './prune_save/cnn_vae_prune_500.ckpt'
	retrain_save_path = './prune_save/cnn_vae_retrain_500.ckpt'


	train_set, test_set, max_min = data_read_7()
	for i in range(10):
		prune_test(i, test_set, ckpt_save_path)


	# masktop5 = []
	# conv1_wa_m = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 25, 26, 27, 30, 34,
	# 			  36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 50, 51, 54, 57, 58, 59, 61, 65, 66, 70, 74, 75, 76,
	# 			  78, 79, 80, 83, 85, 87, 89, 93, 94]
	# for i in range(0, 96, 1):
	# 	if i not in conv1_wa_m:
	# 		test_accuracy, test_accuracy_top5 = prune_test(i, ckpt_save_path)
	# 		if test_accuracy_top5 >= 0.697:
	# 			masktop5.append(i)
	# print(masktop5)

	# weight_prune(ckpt_save_path)
	# evaluation_testset(prune_save_path)
	# for i in range(0, 3, 1):
	# 	retrain(prune_save_path)
	# 	weight_prune(retrain_save_path)
	# 	evaluation_testset(prune_save_path)

	# reader = pywrap_tensorflow.NewCheckpointReader(ckpt_save_path)
	# with tf.Session() as sess:
	# 	all_variables = reader.get_variable_to_shape_map()
	# 	print(all_variables)
	# 	fc1_w = reader.get_tensor("fc1/fc1_w")
	# 	fc2_w = reader.get_tensor("fc2/fc2_w")
	# 	fc3_w = reader.get_tensor("softmax_linear/fc3_w")
	# 	#np.savez("./save_np/fc_w_noprune.npz", fc1_w=fc1_w, fc2_w=fc2_w, fc3_w=fc3_w)
	# 	conv1_w = reader.get_tensor("conv1_w")
	# 	print('conv1_w')
	# 	print(conv1_w.shape)
	# 	print(type(conv1_w))
	# # 	print(fc2_w)
	# # #
	# 	conv2_w = reader.get_tensor("conv2_w")
	# 	print('conv2_w')
	# 	print(conv2_w.shape)
	# 	print(type(conv2_w))
	# 	print(conv2_w)

	# 	print(conv2_w[:, :, :, 0])
	# 	print(conv2_w[:, :, :, 2])
	# 	print(conv2_w[:, :, :, 4])
	# 	print(conv2_w[:, :, :, 5])
	# 	print(conv2_w[:, :, :, 6])
	# 	print(conv2_w[:, :, :, 7])
	# 	print(conv2_w[:, :, :, 9])
	# 	print(conv2_w[:, :, :, 10])
	# 	print(conv2_w[:, :, :, 11])
	# 	print(conv2_w[:, :, :, 12])
	# 	print(conv2_w[:, :, :, 13])
	# 	print(conv2_w[:, :, :, 14])
	# 	print(conv2_w[:, :, :, 15])



