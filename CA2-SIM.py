# coding=utf-8
"""Implementation of CA2-SITIDIM attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import pandas as pd
import scipy.stats as st
from scipy.misc import imread, imsave
import tensorflow as tf
from utils import *

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 4, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 12.0, 'max epsilon.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

# 后续没有用到的全局变量，根据文章中实验章节的参数说明，该参数为DIM算法的参数
#   组合环境中 DIM 方法的输入随机变换概率 = 0.7
tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_integer('sample_num', 4, 'the size of gradient.')

tf.flags.DEFINE_float('sample_variance', 0.1, 'the size of gradient.')

tf.flags.DEFINE_string('checkpoint_path', '/nfs/checkpoints/',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', '/nfs/dataset/ali2019/images1000_val/attack',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir',
                       '/nfs/test/',
                       'Output directory with images.')

FLAGS = tf.flags.FLAGS

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

# select victim model from i3, i4, ir2, r50
model = 'i3'

# settings of cyclical optimization
phases = [3]
phase_step = [4, 4, 8]  # max_iteration=16
phase_num = 3

# TIM
stack_kernel = gkern(11, 3)

# kernel size of cyclical augmentation (self-ensemble policy)
stack_kernel_9 = gkern(9, 3)
stack_kernel_11 = gkern(11, 3)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_50.ckpt'),
    'densenet': os.path.join(FLAGS.checkpoint_path, 'tf-densenet161.ckpt')}

def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def inceptionv3_model(x):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v3, end_points_v3

def inceptionv4_model(x):
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(x), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v4, end_points_v4


def inceptionresnetv2_model(x):
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_res_v2, end_points_res_v2

def resnet50_model(x):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_50(
            input_diversity(x), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_resnet, end_points_resnet

def resnet152_model(x):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
            input_diversity(x), num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_resnet, end_points_resnet

def grad_finish(x, x_ini, one_hot, i, grad):
    # sample number of cyclical augmentation (deviation-augmentation)
    sample_num = FLAGS.sample_num
    return tf.less(i, sample_num)

# DIM算法核心部分：每轮迭代均对输入图像进行随机的RESIZE和PADDING操作
def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    # 获得随机数，后对输入样本进行随机resize变换
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))

    # 此处用到了DIM算法的关键参数prod，还没有理解其作用
    #   随机数<prod，则使用ret=padded，即采用上面的resize&padded后的样本
    #   随机数>=prod，则使用原输入样本，ret=input_tensor
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    # 但是在经过概率选择后，又进行了一次resize变换，重新调整会模型输入的要求大小，即有prob的概率进行变换，1-prod的概率不变换
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # DIM
    return ret

def stop(x, x_ini, y, i, iternum, x_max, x_min, grad):

    return tf.less(i, iternum)

def graph(x, x_ini, y, i, iternum, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    
    tmp = tf.to_float(iternum)
    alpha = eps / tmp
    momentum = FLAGS.momentum
    num_classes = 1001

    logits_ini, end_points_ini = get_logits(x_ini, model)
    pred = tf.argmax(end_points_ini['Predictions'], 1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred[:y.shape[0]] + (1 - first_round) * y
    one_hot = tf.one_hot(y, num_classes)

    num = tf.constant(0)
    _, _, _, _, noise = tf.while_loop(grad_finish, compute_grads,
                                            [x, x_ini, one_hot, num, tf.zeros_like(x)])

    # TIM
    # 这里由于引入TIM算法策略，使用的卷积核尺寸为11，和基础版本的CA2不一致，可以对比基础版本CA2的代码
    # TIM算法的思想是平移不变性攻击，理论上应该在compute_grads()中对样本进行大量的平移变化
    #   但，TIM文章中通过数学证明，对梯度张量进行卷积操作 等价于 大量平移变化，并且可以提升计算效率，所以代码实现如下
    noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    
    x = x + alpha * tf.sign(noise)

    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, x_ini, y, i, iternum, x_max, x_min, noise

def get_logits(x_nes, model):
    logits_v3, end_points_v3 = inceptionv3_model(x_nes)
    logits_v4, end_points_v4 = inceptionv4_model(x_nes)
    logits_v2, end_points_v2 = inceptionresnetv2_model(x_nes)
    logits_50, end_points_50 = resnet50_model(x_nes)

    if(model == 'i3'):
        return logits_v3, end_points_v3
    elif(model == 'i4'):
        return logits_v4, end_points_v4
    elif(model == 'ir2'):
        return logits_v2, end_points_v2
    elif (model == 'r50'):
        return logits_50, end_points_50

def compute_grads(x, x_ini, one_hot, i, grad):
    vector = tf.random_normal(shape=x.shape)

    # cyclical augmentation (deviation-augmentation)
    # sample_variance为偏移距离（文章中的w），可以理解为图像每个像素的偏移量
    # 通过sign函数符号化偏移向量，可以控制每像素偏移值为 -w、0、w
    x_nes = x + FLAGS.sample_variance * tf.sign(vector)

    logits, end_points = get_logits(x_nes, model)
    n = 1

    # SIM
    # 此处为引入SIM算法思想的核心（缩放不变性攻击），即对偏移增强后的样本在每个像素上做数值缩放（每个像素除以一个整数2^i & i=1/2/3，即如下代码中的2/4/8），再计算梯度
    x_nes_2 = 1 / 2 * x_nes
    logits_2, end_points_2 = get_logits(x_nes_2, model)

    x_nes_4 = 1 / 4 * x_nes
    logits_4, end_points_4 = get_logits(x_nes_4, model)

    x_nes_8 = 1 / 8 * x_nes
    logits_8, end_points_8 = get_logits(x_nes_8, model)

    logits = logits + logits_2 + logits_4 + logits_8
    n = n + 3

    # cyclical augmentation (self-ensemble policy)
    x_conv_9 = tf.nn.depthwise_conv2d(x_nes, stack_kernel_9, strides=[1, 1, 1, 1], padding='SAME')
    logits_9, end_points_9 = get_logits(x_conv_9, model)

    x_conv_11 = tf.nn.depthwise_conv2d(x_nes, stack_kernel_11, strides=[1, 1, 1, 1], padding='SAME')
    logits_11, end_points_11 = get_logits(x_conv_11, model)

    logits = logits + logits_9 + logits_11
    n = n + 2

    logits = logits / n
    loss = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0, weights=1.0)
    noise = tf.gradients(loss, x)[0]

    i = tf.add(i, 1)
    grad += noise / FLAGS.sample_num
    return x, x_ini, one_hot, i, grad        

def main(_):
                
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_ini = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        
        iternum = tf.constant(16)
        x_adv, _, _, _, inumber, _, _, noise= tf.while_loop(stop, graph,
                                                    [x_input, x_ini, y, i, iternum, x_max, x_min, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            s2.restore(sess, model_checkpoint_map['inception_v4'])
            s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s4.restore(sess, model_checkpoint_map['resnet_v2'])

            idx = 0
            
            check_or_create_dir(FLAGS.output_dir)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))
                print("output_dir:", FLAGS.output_dir)
                grad_current = np.zeros_like(images)

                # cyclical optimization algorithm
                for k in range(phase_num):
                    iter_num = phase_step[k]
                    adv_images, adv_grad, curr_iternum = sess.run([x_adv, noise, inumber],
                                                                    feed_dict={x_input: images,
                                                                            x_ini: images,
                                                                            grad: grad_current,
                                                                            iternum: iter_num})
                    grad_current = adv_grad

                save_images(adv_images, filenames, FLAGS.output_dir)
                    

if __name__ == '__main__':
    tf.app.run()
