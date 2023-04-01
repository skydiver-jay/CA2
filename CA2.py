# coding=utf-8
"""Implementation of CA2 attack."""

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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 4, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 12.0, 'max epsilon.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

"""
sample_num：
    增加不同类型噪声的数量，属于CA2框架“偏移增强”特性的一个参数。 文章中的k，生成增强数据的数量，也即是随机采样的数量。
"""
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
stack_kernel = gkern(1, 1)

# settings of cyclical optimization
phases = [3]
phase_step = [4, 4, 8]  # max_iteration=16
phase_num = 3

# kernel size of cyclical augmentation (self-ensemble policy)
stack_kernel_9 = gkern(9, 3)
stack_kernel_11 = gkern(11, 3)

model_checkpoint_map = {  # 本地预训练模型文件的存储路径
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
            x, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v3, end_points_v3


def inceptionv4_model(x):
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            x, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v4, end_points_v4


def inceptionresnetv2_model(x):
    """

    Args:
        x: 为inception_resnet_v2()的inputs参数，a 4-D tensor of size [batch_size, height, width, 3]。
            应该就是图像张量。

    Returns: 为inception_resnet_v2()的结果，logits & end_points。
            logits
                相对好理解一些：在机器学习中，logits是指模型的输出，但没有经过softmax或sigmoid等激活函数的变量。它们通常用于计算损失函数，
                而不是直接用于预测。在分类问题中，logits通常是一个向量，每个元素对应一个类别。
            end_points：
                the set of end_points from the inception model。
                还没完全理解。

    """
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            x, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_res_v2, end_points_res_v2


def resnet50_model(x):
    """

    Args:
        x: 为resnet_v2()的inputs参数，A tensor of size [batch, height_in, width_in, channels]。

    Returns: 直接return了resnet_v2()的结果，net & end_points。

            end_points: A dictionary from components of the network to the corresponding
                activation. //在TensorFlow中，end_points是指在模型中的某些位置收集的张量。这些张量可以用于可视化、调试或其他目的。

    注: 如果要集成到Ditto中，这里的常量num_classes（针对ImageNet数据集），需要变为变量。

    """
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_50(
            x, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_resnet, end_points_resnet


def resnet152_model(x):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
            x, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    return logits_resnet, end_points_resnet


def grad_finish(x, x_ini, one_hot, i, grad):
    # sample number of cyclical augmentation (deviation-augmentation)
    sample_num = FLAGS.sample_num
    return tf.less(i, sample_num)


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

    if (model == 'i3'):
        return logits_v3, end_points_v3
    elif (model == 'i4'):
        return logits_v4, end_points_v4
    elif (model == 'ir2'):
        return logits_v2, end_points_v2
    elif (model == 'r50'):
        return logits_50, end_points_50


def compute_grads(x, x_ini, one_hot, i, grad):
    vector = tf.random_normal(shape=x.shape)

    # cyclical augmentation (deviation-augmentation)
    x_nes = x + FLAGS.sample_variance * tf.sign(vector)

    logits, end_points = get_logits(x_nes, model)
    n = 1

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
        x_adv, _, _, _, inumber, _, _, noise = tf.while_loop(stop, graph,
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


"""

the logits outputs of the model:
    在机器学习中，logits是指模型的输出，但没有经过softmax或sigmoid等激活函数的变量。它们通常用于计算损失函数，而不是直接用于预测。在分类问题中，
    logits通常是一个向量，每个元素对应一个类别。

the set of end_points from the inception model:  
    在TensorFlow中，end_points是指在模型中的某些位置收集的张量。这些张量可以用于可视化、调试或其他目的。在Inception模型中，
    end_points包括每个Inception模块的输出，以及最终的logits输出。

"""