# -*- coding: utf-8 -*-
"""
    CBN_VAE
    2019/1/7
    Liu Jianlin
"""

import tensorflow as tf
import numpy as np
import time
import math
from tensorflow.python import pywrap_tensorflow
import scipy.stats as stats
from tensorflow.python.client import timeline
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import flatten

def logistic(x):
    return 1.0 / (1 + tf.exp(-x))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


def max_pool_1x2_with_argmax(input, stride=2):
    '''
    重定义一个最大池化函数，返回最大池化结果以及每个最大值的位置(是个索引，形状和池化结果一致)

    args:
        net:输入数据 形状为[batch,in_height,in_width,in_channels]
        stride：步长，是一个int32类型，注意在最大池化操作中我们设置窗口大小和步长大小是一样的
    '''
    # 使用mask保存每个最大值的位置 这个函数只支持GPU操作
    _, mask = tf.nn.max_pool_with_argmax(input, ksize=[1, 1, stride, 1], strides=[1, 1, stride, 1], padding='SAME')
    # 将反向传播的mask梯度计算停止
    mask = tf.stop_gradient(mask)
    # 计算最大池化操作
    net = tf.nn.max_pool(input, ksize=[1, 1, stride, 1], strides=[1, 1, stride, 1], padding='SAME')
    # 将池化结果和mask返回
    return net, mask


def un_max_pool_1x2(input, mask, stride=2):
    '''
    定义一个反最大池化的函数，找到mask最大的索引，将max的值填到指定位置
    args:
        net:最大池化后的输出，形状为[batch, height, width, in_channels]
        mask：位置索引组数组，形状和net一样
        stride:步长，是一个int32类型，这里就是max_pool_with_argmax传入的stride参数
    '''
    ksize = [1, 1, stride, 1]
    input_shape = input.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(input)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(input, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret


def train(train_set, test_set, max_min, maxepoch=500, save_path="./save/cnn_ae_test_500"):

    global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
    batch_size = 1
    x = tf.placeholder(tf.float32, shape=[batch_size, train_set.shape[1], train_set.shape[2], train_set.shape[3]])  # train set: [289, 1, 120, 1]
    batch_num_train = int(train_set.shape[0] / batch_size)
    batch_num_test = int(test_set.shape[0] / batch_size)

    data = np.zeros([batch_size, train_set.shape[1], train_set.shape[2], train_set.shape[3]], np.float32)
    l2_loss = tf.constant(0.0)

    # encoder
    w_conv1 = weight_variable([1, 12, 1, 24])  # [filter_width, in_channels, out_channels]
    b_conv1 = bias_variable([24])
    conv1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1, 1, 12, 1], padding='SAME') + b_conv1, name="conv1")
    print("conv1 out:" + str(conv1.shape))
    o_conv1 = tf.reshape(conv1, [batch_size, 1, conv1.get_shape()[2].value * conv1.get_shape()[3].value, 1])
    print("o_conv1 out:" + str(o_conv1.shape))

    o_pool1, mask1 = max_pool_1x2_with_argmax(input=o_conv1)
    print("o_pool1 out:" + str(o_pool1.shape))

    w_conv2 = weight_variable([1, 9, 1, 12])  # [filter_width, in_channels, out_channels]
    b_conv2 = bias_variable([12])
    conv2 = tf.nn.relu(tf.nn.conv2d(o_pool1, w_conv2, strides=[1, 1, 9, 1], padding='SAME') + b_conv2, name="conv2")
    print("conv2 out:" + str(conv2.shape))
    o_conv2 = tf.reshape(conv2, [batch_size, 1, conv2.get_shape()[2].value * conv2.get_shape()[3].value, 1])
    print("o_conv2 out:" + str(o_conv2.shape))

    o_pool2, mask2 = max_pool_1x2_with_argmax(input=o_conv2)
    print("o_pool2 out:" + str(o_pool2.shape))
    o_pool3, mask3 = max_pool_1x2_with_argmax(input=o_pool2)
    print("o_pool3 out:" + str(o_pool3.shape))

    w_conv3 = weight_variable([1, 5, 1, 12])  # [filter_width, in_channels, out_channels]
    b_conv3 = bias_variable([12])
    conv3 = tf.nn.relu(tf.nn.conv2d(o_pool3, w_conv3, strides=[1, 1, 5, 1], padding='SAME') + b_conv3, name="conv3")
    print("conv3 out:" + str(conv3.shape))
    o_conv3 = tf.reshape(conv3, [batch_size, 1, conv3.get_shape()[2].value * conv3.get_shape()[3].value, 1])
    print("o_conv3 out:" + str(o_conv3.shape))

    o_pool4, mask4 = max_pool_1x2_with_argmax(input=o_conv3)
    print("o_pool4 out:" + str(o_pool4.shape))

    w_conv4 = weight_variable([1, 3, 1, 12])  # [filter_width, in_channels, out_channels]
    b_conv4 = bias_variable([12])
    conv4 = tf.nn.relu(tf.nn.conv2d(o_pool4, w_conv4, strides=[1, 1, 3, 1], padding='SAME') + b_conv4, name="conv4")
    print("conv4 out:" + str(conv4.shape))
    o_conv4 = tf.reshape(conv4, [batch_size, 1, conv4.get_shape()[2].value * conv4.get_shape()[3].value, 1])
    print("o_conv4 out:" + str(o_conv4.shape))

    o_pool5, mask5 = max_pool_1x2_with_argmax(input=o_conv4)
    print("o_pool5 out:" + str(o_pool5.shape))
    o_pool6, mask6 = max_pool_1x2_with_argmax(input=o_pool5)
    print("o_pool6 out:" + str(o_pool6.shape))

    # full connection1
    w_fc1 = weight_variable([o_pool6.get_shape()[2].value, int(o_pool6.get_shape()[2].value/2)])
    b_fc1 = bias_variable([int(o_pool6.get_shape()[2].value/2)])
    maxpool2_flat = tf.reshape(o_pool6, [-1, o_pool6.get_shape()[2].value])
    o_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, w_fc1) + b_fc1, name="o_fc1")
    print("o_fc1 out:" + str(o_fc1.shape))

    # mean
    w5 = weight_variable([int(o_pool6.get_shape()[2].value/2), 5])
    b5 = bias_variable([5])
    mu_encoder = tf.matmul(o_fc1, w5) + b5
    print(mu_encoder)
    # logvar
    w6 = weight_variable([int(o_pool6.get_shape()[2].value/2), 5])
    b6 = bias_variable([5])
    logvar_encoder = tf.matmul(o_fc1, w6) + b6
    print(logvar_encoder)
    # Sample epsilon
    epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')
    # Sample latent variable
    std_encoder = tf.exp(0.5 * logvar_encoder)
    z = mu_encoder + tf.multiply(epsilon, std_encoder)  # encoder output
    print(z)
    #decoder input
    w7 = weight_variable([5, int(o_pool6.get_shape()[2].value/2)])
    b7 = bias_variable([int(o_pool6.get_shape()[2].value/2)])
    de_input = tf.nn.relu(tf.matmul(z, w7) + b7)
    print(de_input)
    # decoder------------------------------------------------------------------------
    wt_fc1 = weight_variable([int(o_pool6.get_shape()[2].value/2), o_pool6.get_shape()[2].value])
    bt_fc1 = bias_variable([o_pool6.get_shape()[2].value])
    ot_fc1 = tf.nn.relu(tf.matmul(de_input, wt_fc1) + bt_fc1, name="ot_fc1")
    tmaxpool2_flat = tf.reshape(ot_fc1, [batch_size, 1, o_pool6.get_shape()[2].value, 1])
    print("tmaxpool2_flat out:" + str(tmaxpool2_flat.shape))

    o_uppool = un_max_pool_1x2(input=tmaxpool2_flat, mask=mask6)
    print("o_uppool out:" + str(o_uppool.shape))
    o_uppool1 = un_max_pool_1x2(input=o_uppool, mask=mask5)
    print("o_uppool1 out:" + str(o_uppool1.shape))

    r_conv4 = tf.nn.tanh(tf.reshape(o_uppool1, conv4.shape))
    print("r_conv4 out:" + str(r_conv4.shape))

    wt_conv5 = weight_variable([1, 3, 1, 12])  # [filter_width, in_channels, out_channels]
    ot_conv5 = tf.nn.conv2d_transpose(r_conv4, w_conv4, [batch_size, 1, o_pool4.get_shape()[2].value, 1], strides=[1, 1, 3, 1], padding='SAME')
    print("ot_conv5 out:" + str(ot_conv5.shape))

    o_uppool3 = un_max_pool_1x2(input=ot_conv5, mask=mask4)
    print("o_uppool3 out:" + str(o_uppool3.shape))

    r_conv3 = tf.nn.tanh(tf.reshape(o_uppool3, conv3.shape))
    print("r_conv3 out:" + str(r_conv3.shape))

    wt_conv4 = weight_variable([1, 5, 1, 12])  # [filter_width, in_channels, out_channels]
    ot_conv4 = tf.nn.conv2d_transpose(r_conv3, w_conv3, [batch_size, 1, o_pool3.get_shape()[2].value, 1], strides=[1, 1, 5, 1], padding='SAME')
    print("ot_conv4 out:" + str(ot_conv4.shape))

    o_uppool4 = un_max_pool_1x2(input=ot_conv4, mask=mask3)
    print(o_uppool4)
    o_uppool5 = un_max_pool_1x2(input=o_uppool4, mask=mask2)
    print(o_uppool5)

    reshape = tf.nn.tanh(tf.reshape(o_uppool5, conv2.shape))
    print("avg_pool_15 flatten out:" + str(reshape.shape))

    wt_conv3 = weight_variable([1, 9, 1, 12])  # [filter_width, in_channels, out_channels]
    ot_conv3 = tf.nn.conv2d_transpose(reshape, w_conv2, [batch_size, 1, o_pool1.get_shape()[2].value, 1], strides=[1, 1, 9, 1], padding='SAME')
    print(ot_conv3)

    o_uppool6 = un_max_pool_1x2(input=ot_conv3, mask=mask1)
    print(o_uppool6)

    reshape = tf.nn.tanh(tf.reshape(o_uppool6, conv1.shape))
    print("avg_pool_15 flatten out:" + str(reshape.shape))

    wt_conv2 = weight_variable([1, 12, 1, 24])  # [filter_width, in_channels, out_channels]
    de_output = tf.nn.conv2d_transpose(reshape, w_conv1, [batch_size, 1, 120, 1], strides=[1, 1, 12, 1], padding='SAME')
    print(de_output)
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
    train_step = tf.train.AdamOptimizer(lr).minimize(mse, global_step=global_step)
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
                _, train_xloss1, outdata, iters, lr_, prdc, prdd = sess.run([train_step, xloss, de_output, global_step, lr, prd_c, prd_d], feed_dict={x: batch_x_train})

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
            snr = 10 * math.log((1/(prd*prd/10000)), 10)
            print("Step %d global_step %d lr_ %6.6f | Train meanerror %6.6f Test meanerror %6.6f PRD %6.6f SNR %6.6f" % (
                step, iters, lr_, train_errsum / batch_num_train * max_min, test_errsum / batch_num_test * max_min, prd, snr))
        print("over:" + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
        # saver.save(sess, save_path)
        print('Trained Model Saved.')


def evalu(test_set, save_path="./save/cnn_ae_test_500"):
    reader = pywrap_tensorflow.NewCheckpointReader(save_path)
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

    mse = (tf.reduce_sum((de_output - x) * (de_output - x))) / 120
    # loss = mse + lam * l2_loss
    xloss = tf.reduce_mean(abs(de_output - x))


    prd_c = tf.square(xloss)
    prd_d = tf.square(tf.reduce_mean(abs(x)))
    odata = tf.reduce_mean(abs(x))
    rdata = tf.reduce_mean(abs(de_output))

    errsum = 0
    outdata3 = np.zeros([batch_size, batch_num_test * 120], np.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("start:" + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
        for step in range(0, 1):
            train_errsum = 0
            test_errsum = 0
            prd_c_sum = 0
            prd_d_sum = 0

            for offset in range(0, batch_size * batch_num_test, batch_size):
                end = offset + batch_size
                batch_x_test = test_set[offset:end, :, :, :]
                test_xloss1, prdc, prdd, outdata = sess.run([xloss, prd_c, prd_d, de_output], feed_dict={x: batch_x_test})
                test_errsum = test_errsum + test_xloss1
                prd_c_sum = prdc + prd_c_sum
                prd_d_sum = prdd + prd_d_sum
                outdata3[0, offset*120:end*120] = outdata[0, 0, :, 0]
                # for i in range(0,120):
                #     print(batch_x_test[])
            prd = np.sqrt(prd_c_sum / prd_d_sum) * 100
            snr = 10 * math.log((1 / (prd * prd / 10000)), 10)
            print("Test meanerror %6.6f PRD %6.6f" % (test_errsum / batch_num_test * 8.6748982, prd))
        print("over:" + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    # return test_errsum / batch_num_test, prd, snr
    return outdata3


def retrain(train_set, test_set, max_min, maxepoch=500, save_path="./save/cnn_ae_test_500"):
    reader = pywrap_tensorflow.NewCheckpointReader("./save/cnn_ae_test_500")
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

    mse = (tf.reduce_sum((de_output - x) * (de_output - x))) / train_set.shape[2]
    # loss = mse + lam * l2_loss
    xloss = tf.reduce_mean(abs(de_output - x))
    # decay_steps = int(289 * 2.5)
    # lr = tf.train.exponential_decay(0.01, tf.train.get_or_create_global_step(), decay_steps, 0.94, staircase=True)
    lr = tf.cond(tf.less(global_step, int(10000 / batch_size)),
                 lambda: tf.constant(0.001),
                 lambda: tf.cond(tf.less(global_step, int(30000 / batch_size)),
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
            for offset in range(0, batch_size * batch_num_train, batch_size):
                end = offset + batch_size
                batch_x_train = train_set[offset:end, :, :, :]
                # data[0, :, :, :] = train_set[batch*batch_size, :, :, :]
                _, train_xloss1, outdata, iters, lr_ = sess.run([train_step, xloss, de_output, global_step, lr], feed_dict={x: batch_x_train})
                train_errsum = train_errsum + train_xloss1
                # outdata3[0, offset:end] = outdata[0, 0, :, 0]

            for offset in range(0, batch_size * batch_num_test, batch_size):
                end = offset + batch_size
                batch_x_test = test_set[offset:end, :, :, :]
                # data[0, :, :, :] = test_set[batch, :, :, :]
                test_xloss1, prdc, prdd = sess.run([xloss, prd_c, prd_d], feed_dict={x: batch_x_test})
                test_errsum = test_errsum + test_xloss1
                prd_c_sum = prdc + prd_c_sum
                prd_d_sum = prdd + prd_d_sum

            prd = np.sqrt(prd_c_sum / prd_d_sum) * 100
            snr = 10 * math.log((1/(prd*prd/10000)), 10)
            print("Step %d global_step %d lr_ %6.6f | Train meanerror %6.6f Test meanerror %6.6f PRD %6.6f SNR %6.6f" % (
                step, iters, lr_, train_errsum / batch_num_train * max_min, test_errsum / batch_num_test * max_min, prd, snr))
        print("over:" + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
        saver.save(sess, save_path)
        print('Trained Model Saved.')


def trainae(train_set, test_set, max_min, maxepoch=500, save_path="./save/cnn_ae_test_500"):

    global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
    batch_size = 1
    x = tf.placeholder(tf.float32, shape=[batch_size, train_set.shape[1], train_set.shape[2], train_set.shape[3]])  # train set: [289, 1, 120, 1]
    batch_num_train = int(train_set.shape[0] / batch_size)
    batch_num_test = int(test_set.shape[0] / batch_size)

    data = np.zeros([batch_size, train_set.shape[1], train_set.shape[2], train_set.shape[3]], np.float32)

    # encoder
    w_conv1 = weight_variable([1, 12, 1, 24])  # [filter_width, in_channels, out_channels]
    b_conv1 = bias_variable([24])
    conv1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1, name="conv1")
    print("conv1 out:" + str(conv1.shape))

    o_pool1, mask1 = max_pool_1x2_with_argmax(input=conv1)
    print("o_pool1 out:" + str(o_pool1.shape))

    w_conv2 = weight_variable([1, 9, 24, 12])  # [filter_width, in_channels, out_channels]
    b_conv2 = bias_variable([12])
    conv2 = tf.nn.relu(tf.nn.conv2d(o_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2, name="conv2")
    print("conv2 out:" + str(conv2.shape))

    o_pool2, mask2 = max_pool_1x2_with_argmax(input=conv2)
    print("o_pool2 out:" + str(o_pool2.shape))
    o_pool3, mask3 = max_pool_1x2_with_argmax(input=o_pool2)
    print("o_pool3 out:" + str(o_pool3.shape))

    w_conv3 = weight_variable([1, 5, 12, 12])  # [filter_width, in_channels, out_channels]
    b_conv3 = bias_variable([12])
    conv3 = tf.nn.relu(tf.nn.conv2d(o_pool3, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3, name="conv3")
    print("conv3 out:" + str(conv3.shape))

    o_pool4, mask4 = max_pool_1x2_with_argmax(input=conv3)
    print("o_pool4 out:" + str(o_pool4.shape))

    w_conv4 = weight_variable([1, 3, 12, 12])  # [filter_width, in_channels, out_channels]
    b_conv4 = bias_variable([12])
    conv4 = tf.nn.relu(tf.nn.conv2d(o_pool4, w_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4, name="conv4")
    print("conv4 out:" + str(conv4.shape))

    o_pool5, mask5 = max_pool_1x2_with_argmax(input=conv4)
    print("o_pool5 out:" + str(o_pool5.shape))
    o_pool6, mask6 = max_pool_1x2_with_argmax(input=o_pool5)
    print("o_pool6 out:" + str(o_pool6.shape))

    # full connection1
    w_fc1 = weight_variable([o_pool6.get_shape()[2].value * 12, 5])
    b_fc1 = bias_variable([5])
    maxpool2_flat = tf.reshape(o_pool6, [-1, o_pool6.get_shape()[2].value * 12])
    o_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, w_fc1) + b_fc1, name="o_fc1")
    print("o_fc1 out:" + str(o_fc1.shape))

    # decoder------------------------------------------------------------------------
    wt_fc1 = weight_variable([5, o_pool6.get_shape()[2].value * 12])
    bt_fc1 = bias_variable([o_pool6.get_shape()[2].value * 12])
    ot_fc1 = tf.nn.relu(tf.matmul(o_fc1, wt_fc1) + bt_fc1, name="ot_fc1")
    tmaxpool2_flat = tf.reshape(ot_fc1, [batch_size, 1, o_pool6.get_shape()[2].value, 12])
    print("tmaxpool2_flat out:" + str(tmaxpool2_flat.shape))

    o_uppool = un_max_pool_1x2(input=tmaxpool2_flat, mask=mask6)
    print("o_uppool out:" + str(o_uppool.shape))
    o_uppool1 = un_max_pool_1x2(input=o_uppool, mask=mask5)
    print("o_uppool1 out:" + str(o_uppool1.shape))

    wt_conv5 = weight_variable([1, 3, 12, 12])  # [filter_width, in_channels, out_channels]
    ot_conv5 = tf.nn.conv2d_transpose(o_uppool1, wt_conv5, [batch_size, 1, o_pool4.get_shape()[2].value, 12], strides=[1, 1, 1, 1], padding='SAME')
    print("ot_conv5 out:" + str(ot_conv5.shape))

    o_uppool3 = un_max_pool_1x2(input=ot_conv5, mask=mask4)
    print("o_uppool3 out:" + str(o_uppool3.shape))

    w_fc2 = weight_variable([16 * 12, 15*12])
    b_fc2 = bias_variable([15*12])
    maxpool2_flat = tf.reshape(o_uppool3, [-1, 16 * 12])
    o_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, w_fc2) + b_fc2, name="o_1fc1")
    # o_uppool3 = tf.reshape(o_fc1, [1, 1, 15, 12])
	#
	#
    # wt_conv4 = weight_variable([1, 5, 12, 12])  # [filter_width, in_channels, out_channels]
    # ot_conv4 = tf.nn.conv2d_transpose(o_uppool3, wt_conv4, [batch_size, 1, o_pool3.get_shape()[2].value, 12], strides=[1, 1, 1, 1], padding='SAME')
    # print("ot_conv4 out:" + str(ot_conv4.shape))
	#
    # o_uppool4 = un_max_pool_1x2(input=ot_conv4, mask=mask3)
    # print(o_uppool4)
    # o_uppool5 = un_max_pool_1x2(input=o_uppool4, mask=mask2)
    # print(o_uppool5)
	#
    # wt_conv3 = weight_variable([1, 9, 24, 12])  # [filter_width, in_channels, out_channels]
    # ot_conv3 = tf.nn.conv2d_transpose(o_uppool5, wt_conv3, [batch_size, 1, o_pool1.get_shape()[2].value, 24], strides=[1, 1, 1, 1], padding='SAME')
    # print(ot_conv3)
	#
    # o_uppool6 = un_max_pool_1x2(input=ot_conv3, mask=mask1)
    # print(o_uppool6)
	#
    # wt_conv2 = weight_variable([1, 12, 1, 24])  # [filter_width, in_channels, out_channels]
    # de_output = tf.nn.conv2d_transpose(o_uppool6, wt_conv2, [batch_size, 1, 120, 1], strides=[1, 1, 1, 1], padding='SAME')
    # print(de_output)
	#
    # mse = (tf.reduce_sum((de_output - x) * (de_output - x))) / train_set.shape[2]
    # # loss = mse + lam * l2_loss
    # xloss = tf.reduce_mean(abs(de_output - x))
    # # decay_steps = int(289 * 2.5)
    # # lr = tf.train.exponential_decay(0.01, tf.train.get_or_create_global_step(), decay_steps, 0.94, staircase=True)
    # lr = tf.cond(tf.less(global_step, int(10000 / batch_size)),
    #              lambda: tf.constant(0.001),
    #              lambda: tf.cond(tf.less(global_step, int(30000 / batch_size)),
    #                              lambda: tf.constant(0.0001),
    #                              lambda: tf.constant(0.00001)))
    # train_step = tf.train.AdamOptimizer(lr).minimize(mse, global_step=global_step)
    # # # add Saver ops
    # # saver = tf.train.Saver({'w_conv1': w_conv1, 'b_conv1': b_conv1,
    # #                         'w_conv2': w_conv2, 'b_conv2': b_conv2,
    # #                         'w_conv3': w_conv3, 'b_conv3': b_conv3,
    # #                         'w_conv4': w_conv4, 'b_conv4': b_conv4,
    # #                         'w_fc1': w_fc1, 'b_fc1': b_fc1,
    # #                         'w5': w5, 'b5': b5,
    # #                         'w6': w6, 'b6': b6,
    # #                         'w7': w7, 'b7': b7,
    # #                         'wt_fc1': wt_fc1, 'bt_fc1': bt_fc1
	# #
    # #                         })
    # # saver = tf.train.Saver()
	#
    # prd_c = tf.square(xloss)
    # prd_d = tf.square(tf.reduce_mean(abs(x)))
    # odata = tf.reduce_mean(abs(x))
    # rdata = tf.reduce_mean(abs(de_output))
	#
    # errsum = 0
    # max_min = max_min[0] - max_min[1]
    # outdata3 = np.zeros([batch_size, batch_num_train * 120], np.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("start:" + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
        for step in range(0, 10):
            train_errsum = 0
            test_errsum = 0
            prd_c_sum = 0
            prd_d_sum = 0
            starttime = time.time()
            for i in range(100):
                for offset in range(0, batch_size * batch_num_train, batch_size):
                    end = offset + batch_size
                    batch_x_train = train_set[offset:end, :, :, :]
                    # data[0, :, :, :] = train_set[batch*batch_size, :, :, :]
                    # _, train_xloss1, outdata, iters, lr_ = sess.run([train_step, xloss, de_output, global_step, lr], feed_dict={x: batch_x_train})
                    _ = sess.run([o_fc1], feed_dict={x: batch_x_train})
                    # train_errsum = train_errsum + train_xloss1
                    # outdata3[0, offset:end] = outdata[0, 0, :, 0]
            print("Train Time: " + str(int(time.time() - starttime)) + 's')
            # starttime = time.time()
            # for offset in range(0, batch_size * batch_num_test, batch_size):
            #     end = offset + batch_size
            #     batch_x_test = test_set[offset:end, :, :, :]
            #     # data[0, :, :, :] = test_set[batch, :, :, :]
            #     test_xloss1, prdc, prdd = sess.run([xloss, prd_c, prd_d], feed_dict={x: batch_x_test})
            #     test_errsum = test_errsum + test_xloss1
            #     prd_c_sum = prdc + prd_c_sum
            #     prd_d_sum = prdd + prd_d_sum
			#
            # prd = np.sqrt(prd_c_sum / prd_d_sum) * 100
            # snr = 10 * math.log((1/(prd*prd/10000)), 10)
            # print("Step %d global_step %d lr_ %6.6f | Train meanerror %6.6f Test meanerror %6.6f PRD %6.6f SNR %6.6f" % (
            #     step, iters, lr_, train_errsum / batch_num_train * max_min, test_errsum / batch_num_test * max_min, prd, snr))
        print("over:" + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
        # saver.save(sess, save_path)
        print('Trained Model Saved.')


