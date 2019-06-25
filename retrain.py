# -*- coding: utf-8 -*-
"""
    Retrain
    2019/04/17
    Liu Jianlin
"""
from data import *
from cbn_vae import *
from tensorflow.python import pywrap_tensorflow
import numpy as np
import time


def weight_prune(prune_file, prune_save_path='./prune_save/cnn_vae_prune_500.ckpt'):
	with tf.Session() as sess:
		reader = pywrap_tensorflow.NewCheckpointReader(prune_file)

		conv1_wa = reader.get_tensor("w_conv1")
		conv1_wa_m = [0,3,5,9,7,11,13,15,17,18,21,23]
		conv1_wa_m = [0, 3, 5, 9, 7, 11, 13,15,17,18,21,23]
		print(len(conv1_wa_m))
		for i in conv1_wa_m:
			conv1_wa[:, :, :, i] = 0

		conv2_wa = reader.get_tensor("w_conv2")
		conv2_wa_m = [1,3,4,5,10,11,0,2,6,7,8]
		conv2_wa_m = [1,3,4,5,10,11]
		print(len(conv2_wa_m))
		for i in conv2_wa_m:
			conv2_wa[:, :, :, i] = 0

		conv3_wa = reader.get_tensor("w_conv3")
		conv3_wa_m = [5,7,11,0,1,2,3,8,9]
		conv3_wa_m = [5,7,11]
		print(len(conv3_wa_m))
		for i in conv3_wa_m:
			conv3_wa[:, :, :, i] = 0

		conv4_wa = reader.get_tensor("w_conv4")
		conv4_wa_m = [0,1,3,5,6,7,9,10,11]
		conv4_wa_m = [0,1,3,5,6,7,9,10,11]
		print(len(conv4_wa_m))
		for i in conv4_wa_m:
			conv4_wa[:, :, :, i] = 0

		conv1_w = tf.Variable(tf.convert_to_tensor(conv1_wa))
		conv1_b = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv1")))
		conv2_w = tf.Variable(tf.convert_to_tensor(conv2_wa))
		conv2_b = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv2")))
		conv3_w = tf.Variable(tf.convert_to_tensor(conv3_wa))
		conv3_b = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv3")))
		conv4_w = tf.Variable(tf.convert_to_tensor(conv4_wa))
		conv4_b = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv4")))

		fc1_wa = reader.get_tensor("w_fc1")
		fc1_w_mask = np.load("./save_kmeans/w_fc1_clusterAssment.npz")
		fc1_w_clusterAssment = np.mat(fc1_w_mask['fc1_w_clusterAssment'])
		fc1_w_m = fc1_w_clusterAssment.reshape([54, 27])
		fc1_prune_num = 0
		for i in range(0, fc1_w_m.shape[0]):
			for j in range(0, fc1_w_m.shape[1]):
				if (abs(fc1_w_m[i, j]) == 0 or abs(fc1_w_m[i, j]) == 9 or abs(fc1_w_m[i, j]) == 8 or abs(fc1_w_m[i, j]) == 1 or abs(fc1_w_m[i, j]) == 3 or abs(fc1_w_m[i, j]) == 2 or abs(fc1_w_m[i, j]) == 5 or abs(fc1_w_m[i, j]) == 7 or abs(fc1_w_m[i, j]) == 4):
					fc1_wa[i, j] = 0.
					fc1_prune_num += 1
		print(fc1_prune_num)
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
				if (abs(fct1_w_m[i, j]) == 0 or abs(fct1_w_m[i, j]) == 1 or abs(fct1_w_m[i, j]) == 3 or abs(fct1_w_m[i, j]) == 7 or abs(fct1_w_m[i, j]) == 6 or abs(fct1_w_m[i, j]) == 4 or abs(fct1_w_m[i, j]) == 2 or abs(fct1_w_m[i, j]) == 9):
					fct1_wa[i, j] = 0.
					fct1_prune_num += 1
		print(fct1_prune_num)
		wt_fc1 = tf.Variable(tf.convert_to_tensor(fct1_wa))

		# wt_fc1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("wt_fc1")))
		bt_fc1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("bt_fc1")))

		sess.run(tf.global_variables_initializer())

		# add Saver ops
		saver = tf.train.Saver({'w_conv1': conv1_w, 'b_conv1': conv1_b,
								'w_conv2': conv2_w, 'b_conv2': conv2_b,
								'w_conv3': conv3_w, 'b_conv3': conv3_b,
								'w_conv4': conv4_w, 'b_conv4': conv4_b,
								'w_fc1': w_fc1, 'b_fc1': b_fc1,
								'w5': w5, 'b5': b5,
								'w6': w6, 'b6': b6,
								'w7': w7, 'b7': b7,
								'wt_fc1': wt_fc1, 'bt_fc1': bt_fc1})

		saver.save(sess, prune_save_path)
		print('The weight after prune is saved.')


def retrain(train_set, test_set, max_min, maxepoch=500, ckpt_path='./prune_save/cnn_vae_prune_500.ckpt', save_path='./prune_save/cnn_vae_retrain_500.ckpt'):
	tf.reset_default_graph()
	reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
	w_conv1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w_conv1")))
	b_conv1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv1")))
	w_conv2 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w_conv2")))
	b_conv2 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv2")))
	w_conv3 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w_conv3")))
	b_conv3 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv3")))
	w_conv4 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w_conv4")))
	b_conv4 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_conv4")))
	w_fc1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w_fc1")))
	b_fc1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b_fc1")))
	w5 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w5")))
	b5 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b5")))
	w6 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w6")))
	b6 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b6")))
	w7 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("w7")))
	b7 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("b7")))
	wt_fc1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("wt_fc1")))
	bt_fc1 = tf.Variable(tf.convert_to_tensor(reader.get_tensor("bt_fc1")))
	global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
	batch_size = 1
	x = tf.placeholder(tf.float32, shape=[batch_size, train_set.shape[1], train_set.shape[2],
										  train_set.shape[3]])  # train set: [289, 1, 120, 1]
	batch_num_train = int(train_set.shape[0] / batch_size)
	batch_num_test = int(test_set.shape[0] / batch_size)

	data = np.zeros([batch_size, train_set.shape[1], train_set.shape[2], train_set.shape[3]], np.float32)
	l2_loss = tf.constant(0.0)

	# encoder
	conv1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1, 1, 12, 1], padding='SAME') + b_conv1, name="conv1")
	# print("conv1 out:" + str(conv1.shape))
	o_conv1 = tf.reshape(conv1, [batch_size, 1, conv1.get_shape()[2].value * conv1.get_shape()[3].value, 1])
	# print("o_conv1 out:" + str(o_conv1.shape))

	o_pool1, mask1 = max_pool_1x2_with_argmax(input=o_conv1)
	# print("o_pool1 out:" + str(o_pool1.shape))

	conv2 = tf.nn.relu(tf.nn.conv2d(o_pool1, w_conv2, strides=[1, 1, 9, 1], padding='SAME') + b_conv2, name="conv2")
	# print("conv2 out:" + str(conv2.shape))
	o_conv2 = tf.reshape(conv2, [batch_size, 1, conv2.get_shape()[2].value * conv2.get_shape()[3].value, 1])
	# print("o_conv2 out:" + str(o_conv2.shape))

	o_pool2, mask2 = max_pool_1x2_with_argmax(input=o_conv2)
	# print("o_pool2 out:" + str(o_pool2.shape))
	o_pool3, mask3 = max_pool_1x2_with_argmax(input=o_pool2)
	# print("o_pool3 out:" + str(o_pool3.shape))

	conv3 = tf.nn.relu(tf.nn.conv2d(o_pool3, w_conv3, strides=[1, 1, 5, 1], padding='SAME') + b_conv3, name="conv3")
	# print("conv3 out:" + str(conv3.shape))
	o_conv3 = tf.reshape(conv3, [batch_size, 1, conv3.get_shape()[2].value * conv3.get_shape()[3].value, 1])
	# print("o_conv3 out:" + str(o_conv3.shape))

	o_pool4, mask4 = max_pool_1x2_with_argmax(input=o_conv3)
	# print("o_pool4 out:" + str(o_pool4.shape))

	conv4 = tf.nn.relu(tf.nn.conv2d(o_pool4, w_conv4, strides=[1, 1, 3, 1], padding='SAME') + b_conv4, name="conv4")
	# print("conv4 out:" + str(conv4.shape))
	o_conv4 = tf.reshape(conv4, [batch_size, 1, conv4.get_shape()[2].value * conv4.get_shape()[3].value, 1])
	# print("o_conv4 out:" + str(o_conv4.shape))

	o_pool5, mask5 = max_pool_1x2_with_argmax(input=o_conv4)
	# print("o_pool5 out:" + str(o_pool5.shape))
	o_pool6, mask6 = max_pool_1x2_with_argmax(input=o_pool5)
	# print("o_pool6 out:" + str(o_pool6.shape))

	# full connection1
	maxpool2_flat = tf.reshape(o_pool6, [-1, o_pool6.get_shape()[2].value])
	o_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, w_fc1) + b_fc1, name="o_fc1")
	# print("o_fc1 out:" + str(o_fc1.shape))

	# mean
	mu_encoder = tf.matmul(o_fc1, w5) + b5
	# print(mu_encoder)
	# logvar
	logvar_encoder = tf.matmul(o_fc1, w6) + b6
	# print(logvar_encoder)
	# Sample epsilon
	epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')
	# Sample latent variable
	std_encoder = tf.exp(0.5 * logvar_encoder)
	z = mu_encoder + tf.multiply(epsilon, std_encoder)  # encoder output
	# print(z)
	# decoder input

	de_input = tf.nn.relu(tf.matmul(z, w7) + b7)
	# print(de_input)
	# decoder------------------------------------------------------------------------

	ot_fc1 = tf.nn.relu(tf.matmul(de_input, wt_fc1) + bt_fc1, name="ot_fc1")
	tmaxpool2_flat = tf.reshape(ot_fc1, [batch_size, 1, o_pool6.get_shape()[2].value, 1])
	# print("tmaxpool2_flat out:" + str(tmaxpool2_flat.shape))

	o_uppool = un_max_pool_1x2(input=tmaxpool2_flat, mask=mask6)
	# print("o_uppool out:" + str(o_uppool.shape))
	o_uppool1 = un_max_pool_1x2(input=o_uppool, mask=mask5)
	# print("o_uppool1 out:" + str(o_uppool1.shape))

	r_conv4 = tf.nn.tanh(tf.reshape(o_uppool1, conv4.shape))
	# print("r_conv4 out:" + str(r_conv4.shape))

	ot_conv5 = tf.nn.conv2d_transpose(r_conv4, w_conv4, [batch_size, 1, o_pool4.get_shape()[2].value, 1],
									  strides=[1, 1, 3, 1], padding='SAME')
	# print("ot_conv5 out:" + str(ot_conv5.shape))

	o_uppool3 = un_max_pool_1x2(input=ot_conv5, mask=mask4)
	# print("o_uppool3 out:" + str(o_uppool3.shape))

	r_conv3 = tf.nn.tanh(tf.reshape(o_uppool3, conv3.shape))
	# print("r_conv3 out:" + str(r_conv3.shape))

	ot_conv4 = tf.nn.conv2d_transpose(r_conv3, w_conv3, [batch_size, 1, o_pool3.get_shape()[2].value, 1],
									  strides=[1, 1, 5, 1], padding='SAME')
	# print("ot_conv4 out:" + str(ot_conv4.shape))

	o_uppool4 = un_max_pool_1x2(input=ot_conv4, mask=mask3)
	# print(o_uppool4)
	o_uppool5 = un_max_pool_1x2(input=o_uppool4, mask=mask2)
	# print(o_uppool5)

	reshape = tf.nn.tanh(tf.reshape(o_uppool5, conv2.shape))
	# print("avg_pool_15 flatten out:" + str(reshape.shape))

	ot_conv3 = tf.nn.conv2d_transpose(reshape, w_conv2, [batch_size, 1, o_pool1.get_shape()[2].value, 1],
									  strides=[1, 1, 9, 1], padding='SAME')
	# print(ot_conv3)

	o_uppool6 = un_max_pool_1x2(input=ot_conv3, mask=mask1)
	# print(o_uppool6)

	reshape = tf.nn.tanh(tf.reshape(o_uppool6, conv1.shape))
	# print("avg_pool_15 flatten out:" + str(reshape.shape))

	de_output = tf.nn.conv2d_transpose(reshape, w_conv1, [batch_size, 1, 120, 1], strides=[1, 1, 12, 1], padding='SAME')
	# print(de_output)
	#
	mse = (tf.reduce_sum((de_output - x) * (de_output - x))) / train_set.shape[2]
	# loss = mse + lam * l2_loss
	xloss = tf.reduce_mean(abs(de_output - x))
	# decay_steps = int(289 * 2.5)
	# lr = tf.train.exponential_decay(0.01, tf.train.get_or_create_global_step(), decay_steps, 0.94, staircase=True)
	lr = tf.cond(tf.less(global_step, int(10000 / 1)),
				 lambda: tf.constant(0.001),
				 lambda: tf.cond(tf.less(global_step, int(30000 / 1)),
								 lambda: tf.constant(0.0001),
								 lambda: tf.constant(0.00001)))
	train_step = tf.train.AdamOptimizer(0.0001).minimize(mse, global_step=global_step)
	# # add Saver ops
	saver = tf.train.Saver({'w_conv1': w_conv1, 'b_conv1': b_conv1,
							'w_conv2': w_conv2, 'b_conv2': b_conv2,
							'w_conv3': w_conv3, 'b_conv3': b_conv3,
							'w_conv4': w_conv4, 'b_conv4': b_conv4,
							'w_fc1': w_fc1, 'b_fc1': b_fc1,
							'w5': w5, 'b5': b5,
							'w6': w6, 'b6': b6,
							'w7': w7, 'b7': b7,
							'wt_fc1': wt_fc1, 'bt_fc1': bt_fc1

							})
	# saver = tf.train.Saver()

	prd_c = tf.square(xloss)
	prd_d = tf.square(tf.reduce_mean(abs(x)))
	odata = tf.reduce_mean(abs(x))
	rdata = tf.reduce_mean(abs(de_output))

	errsum = 0
	max_min = max_min[0] - max_min[1]
	outdata3 = np.zeros([batch_size, batch_num_train * 120], np.float32)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("start:" + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
		for step in range(0, maxepoch):
			train_errsum = 0
			test_errsum = 0
			prd_c_sum = 0
			prd_d_sum = 0
			starttime = time.time()

			for offset in range(0, batch_size * batch_num_train, batch_size):
				end = offset + batch_size
				batch_x_train = train_set[offset:end, :, :, :]
				# data[0, :, :, :] = train_set[batch*batch_size, :, :, :]
				_, train_xloss1, outdata, iters, lr_, prdc, prdd = sess.run(
					[train_step, xloss, de_output, global_step, lr, prd_c, prd_d], feed_dict={x: batch_x_train})

				train_errsum = train_errsum + train_xloss1
			# outdata3[0, offset:end] = outdata[0, 0, :, 0]
			# print("Train Time: " + str(int(time.time() - starttime)) + 's')
			# starttime = time.time()
			for offset in range(0, batch_size * batch_num_test, batch_size):
				end = offset + batch_size
				batch_x_test = test_set[offset:end, :, :, :]
				# data[0, :, :, :] = test_set[batch, :, :, :]
				test_xloss1, prdc, prdd = sess.run([xloss, prd_c, prd_d], feed_dict={x: batch_x_test})
				test_errsum = test_errsum + test_xloss1
				prd_c_sum = prdc + prd_c_sum
				prd_d_sum = prdd + prd_d_sum

			prd = np.sqrt(prd_c_sum / prd_d_sum) * 100
			snr = 10 * math.log((1 / (prd * prd / 10000)), 10)
			print(
				"Step %d global_step %d lr_ %6.6f | Train meanerror %6.6f Test meanerror %6.6f PRD %6.6f SNR %6.6f" % (
					step, iters, lr_, train_errsum / batch_num_train * max_min, test_errsum / batch_num_test * max_min,
					prd, snr))
		print("over:" + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
		saver.save(sess, save_path)
		print('RETrained Model Saved.')