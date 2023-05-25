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

# 同时构造对抗样本的数量
tf.flags.DEFINE_integer('batch_size', 4, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 12.0, 'max epsilon.')

# 动量衰减系数/动量衰减权重
tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

# 由于当前代码中的本地白盒模型设置为I3，所以image_width、image_height、image_resize几个参数与为适配I3模型的值
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

# 后续没有用到的全局变量，根据文章中实验章节的参数说明，该参数为DIM算法的参数
#   组合环境中 DIM 方法的输入随机变换概率 = 0.7
tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

"""
sample_num: 属于CA2框架“偏移增强”特性的一个参数。 文章中的k，生成偏移增强样本的数量，也即是随机采样的数量。
sample_variance: 为偏移距离，即文章中的ω，文章中实验设置为0.05，此处设置为0.1，待确认哪个是默认推荐配置？？
"""
tf.flags.DEFINE_integer('sample_num', 4, 'the number of samples for SA.')

tf.flags.DEFINE_float('sample_variance', 0.1, '...')

"""
checkpoint_path: 配置本地用于存放预训练模型文件的根路径
    模型文件的绝对路径，由checkpoint_path和文件名拼接，配置在后续代码model_checkpoint_map变量中
"""
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

# 配置本地白盒模型网络类型
# 如果想和文章中“集成模型攻击实验”一样，同时攻击4中本地预训练模型，该如何配置？？
# select victim model from i3, i4, ir2, r50
model = 'i3'


"""
循环优化策略配置：
    如下配置表示，将迭代总数为16的过程，分为3个阶段进行循环优化，各个阶段的迭代次数为4、4、8。
    phases 和 phase_num看上去都是定义的循环优化阶段数，但实际后续代码中没有使用到phases，只用到了phase_num
"""
# settings of cyclical optimization
phases = [3]  # 项目中实际没有使用的变量
phase_step = [4, 4, 8]  # max_iteration=16
phase_num = 3

"""
配置卷积核尺寸分别为9和11的两个高斯卷积矩阵
文章中结论：选择卷积核尺寸集合为 {1,9,11}，默认情况下建议使用此配置；当卷积核尺寸为1时，实则为原模型不变，这个在后续全局梯度计算相关代码中有体现
"""
# kernel size of cyclical augmentation (self-ensemble policy)
stack_kernel = gkern(1, 1)
stack_kernel_9 = gkern(9, 3)
stack_kernel_11 = gkern(11, 3)

# 本地预训练模型文件的绝对路径
model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),   # 即i3
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),   # 即i4
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),  # 即ir2
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_50.ckpt'),  # 即r50
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

    Returns:
        为inception_resnet_v2()的结果，logits & end_points。
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

    Returns:
        直接return了resnet_v2()的结果，net & end_points。

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
    """用于偏移增强策略中进行遍历循环退出判断

    Args:
        i: 应用偏移增强策略时，损失函数计算需要遍历k个增强样本，i为遍历循环控制变量

    Returns:
        如果i<sample_num(即k)，返回True；否则返回False
    """
    # sample number of cyclical augmentation (deviation-augmentation)
    sample_num = FLAGS.sample_num
    return tf.less(i, sample_num)


def stop(x, x_ini, y, i, iternum, x_max, x_min, grad):
    """单纯用于迭代退出判断的函数，返回True循环继续，
        结合后面代码分析，stop()用于循环优化中每个阶段的迭代退出判断
    Args:
        i: 应用循环优化策略时，i为当前阶段的迭代数控制变量
        iternum: 为循环优化当前阶段的总迭代数

    Returns:
        如果i<iternum，返回True，迭代继续；否则返回False，迭代退出
    """
    return tf.less(i, iternum)


def graph(x, x_ini, y, i, iternum, x_max, x_min, grad):
    """当前推测为样本更新代码，为循环优化中单次迭代的样本更新代码
    Args:
        x: 当前循环优化阶段当前迭代的样本x，在当前阶段中会更新
        x_ini: 当前循环优化阶段的初始输入样本，不更新，实际上在整个全局迭代中都保持不变
        i: 应用循环优化策略时，i为当前阶段的迭代数控制变量
        iternum: 为循环优化当前阶段的总迭代数
        x_max: x_max和x_min在迭代中不再变化，用于控制循环优化过程中的样本中间结果，避免样本中间值超过全局设置的最大扰动约束；在main()中根据初始图像张量即最大扰动约束进行设置
        x_min: 同上
        grad: 上一次迭代的循环优化梯度g

    Returns:
        元组--x, x_ini, y, i, iternum, x_max, x_min, noise
    """

    # 设置最大扰动约束：2*12/255 ？， 文章中不是提到是 12/255吗？
    eps = 2.0 * FLAGS.max_epsilon / 255.0

    # 设置当前阶段的步长，即文章中的 α = e/N(t)
    tmp = tf.to_float(iternum)
    alpha = eps / tmp

    # 动量衰减系数/动量衰减权重
    momentum = FLAGS.momentum
    num_classes = 1001

    # 获取初始图像张量的模型输出，model为模型网络类型配置，是一个全局配置
    logits_ini, end_points_ini = get_logits(x_ini, model)

    # ？？？
    pred = tf.argmax(end_points_ini['Predictions'], 1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred[:y.shape[0]] + (1 - first_round) * y
    one_hot = tf.one_hot(y, num_classes)

    # 初始化偏移增强策略中的循环控制变量
    num = tf.constant(0)
    # 应用自增强策略，获得当前循环优化阶段当前迭代的noise，此处noise即全局损失函数梯度值
    _, _, _, _, noise = tf.while_loop(grad_finish, compute_grads,
                                      [x, x_ini, one_hot, num, tf.zeros_like(x)])

    # ？？为什么这里的扰动需要做一次尺寸为1的高斯卷积运算？？
    noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

    # 和cleverhans中的MIM算法实现一样，使用的是reduce_mean()，而非MIM算法原文中的计算L1范数 \
    #   此处noise最终为根据动量优化策略计算当前迭代的循环优化梯度g，momentum就是速度衰减权重μ
    # 循环优化的一个关键点：知识蒸馏，存在一个学习权重β参数，在每个循环优化阶段的初始迭代中，前序迭代的g需要乘以β后再进行后续的迭代累加 \
    #   但是由于文中实验表明，优选的β值为1.0，所以这里在代码实现中没有体现出来这个关键点。后续迁移至tf2，建议体现出该配置，也便于理解算法思想。
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise

    # 更新当前循环优化阶段当前迭代的样本x
    # 循环优化的另一个关键点：在每个循环优化阶段的初始迭代中，计算得到的循环优化梯度g重新施加在初始图像张量x上 \
    #   这一点在该函数中未体现，该函数只呈现单个迭代逻辑，总是将当前迭代计算的g施加在入参样本x上，至于循环优化关键点由main()中循环优化阶段 \
    #   的for循环中特别注意每个循环传入的入参x必须为原始图像张量
    x = x + alpha * tf.sign(noise)

    # 控制样本中间结果不超出全局最大扰动约束
    x = tf.clip_by_value(x, x_min, x_max)
    # 循环优化当前阶段循环控制变量加1
    i = tf.add(i, 1)

    return x, x_ini, y, i, iternum, x_max, x_min, noise


def get_logits(x_nes, model):
    """根据配置的本地白盒模型网络类型，返回相应模型对于x_nes的输出
    Args:
        x_nes: 当前样本张量
        model: 模型网络类型

    Returns:
        返回模型输出元组 logits, end_points, 但end_points在整个CA2项目中并没有使用到
    """
    logits_v3, end_points_v3 = inceptionv3_model(x_nes)
    logits_v4, end_points_v4 = inceptionv4_model(x_nes)
    logits_v2, end_points_v2 = inceptionresnetv2_model(x_nes)
    logits_50, end_points_50 = resnet50_model(x_nes)
    # 可讲模型计算合并入if分支，提升性能
    if (model == 'i3'):
        return logits_v3, end_points_v3
    elif (model == 'i4'):
        return logits_v4, end_points_v4
    elif (model == 'ir2'):
        return logits_v2, end_points_v2
    elif (model == 'r50'):
        return logits_50, end_points_50


def compute_grads(x, x_ini, one_hot, i, grad):
    """CA2框架策略（偏移增强 & 虚拟集成）下的全局梯度计算

    Args:
        x: 当前迭代的图像张量
        x_ini: ??
        one_hot: ??
        i: 应用偏移增强策略时，损失函数计算需要遍历k个增强样本，i为遍历循环控制变量
        grad: 采用偏移增强和虚拟集成策略的累加梯度

    Returns:
        各入参更新后的元组
    """
    # 随机采样得到偏移向量
    vector = tf.random_normal(shape=x.shape)

    # cyclical augmentation (deviation-augmentation)
    # sample_variance为偏移距离（文章中的w），可以理解为图像每个像素的偏移量
    # 通过sign函数符号化偏移向量，可以控制每像素偏移值为 -w、0、w
    x_nes = x + FLAGS.sample_variance * tf.sign(vector)

    # 根据配置的本地白盒模型网络类型，返回相应模型对于x_nes的输出
    logits, end_points = get_logits(x_nes, model)

    # ？？
    n = 1

    # cyclical augmentation (self-ensemble policy)
    # 虚拟集成策略的实施，分别使用尺寸为9和11的高斯卷积核，在same模式下进行卷积计算，same模型为了保证输出后张量shape不变
    # 虽然文章推荐使用局卷积核尺寸为{1,9,11}，但为了代码更为灵活便于调整，后续可将此卷积核尺寸集合配置变为动态可配
    x_conv_9 = tf.nn.depthwise_conv2d(x_nes, stack_kernel_9, strides=[1, 1, 1, 1], padding='SAME')
    logits_9, end_points_9 = get_logits(x_conv_9, model)

    x_conv_11 = tf.nn.depthwise_conv2d(x_nes, stack_kernel_11, strides=[1, 1, 1, 1], padding='SAME')
    logits_11, end_points_11 = get_logits(x_conv_11, model)

    # 由于卷积核尺寸为1时，虚拟模型和原模型一致，所以这里直接使用原输出进行累加
    logits = logits + logits_9 + logits_11
    n = n + 2

    # 多个虚拟模型输出的均值，用于后续计算当前偏移增强样本的损失函数
    logits = logits / n
    loss = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0, weights=1.0)
    noise = tf.gradients(loss, x)[0]

    # 循环控制+1
    i = tf.add(i, 1)
    # 当前偏移增强样本损失函数的梯度值 / k ，再进行循环累加 <=等价于=> 累加梯度求平均
    grad += noise / FLAGS.sample_num
    return x, x_ini, one_hot, i, grad


def main(_):

    # 设置最大扰动约束：2*12/255 ？， 文章中不是提到是 12/255吗？
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        # 关于placeholder的说明，参考https://blog.csdn.net/kdongyi/article/details/82343712
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_ini = tf.placeholder(tf.float32, shape=batch_shape)
        # 利用clip_by_value函数修边（控制边界为 -1 或 1），获得正负方向施加最大扰动后的图像张量
        # x_max和x_min在迭代中不再变化，用于控制循环优化过程中的样本中间结果，避免样本中间值超过全局设置的最大扰动约束
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        # np.zeros([batch_size]) 为初始化一个长度为batch_size的一维全0.向量，如：
        #   np.zeros([3])
        #   Out[3]: array([0., 0., 0.])
        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        # 初始化梯度值为全0
        grad = tf.zeros(shape=batch_shape)

        # 设置总迭代次数，为各循环优化阶段迭达数（4,4,8）之和
        iternum = tf.constant(16)

        # stop()为全局迭达退出判断逻辑
        # graph()为样本构造逻辑
        # x_adv为构造的对抗样本
        # [x_input, x_ini, y, i, iternum, x_max, x_min, grad]为循环变量，但其中只有i和iternum用于判断迭代是否退出
        x_adv, _, _, _, inumber, _, _, noise = tf.while_loop(stop, graph,
                                                             [x_input, x_ini, y, i, iternum, x_max, x_min, grad])

        # Run computation
        # 实例化多个Saver实例，用于后面在session中加载模型；关于tf.train.Saver()可参考https://zhuanlan.zhihu.com/p/474395332
        # 关于slim.get_model_variables()可参考https://cloud.tencent.com/developer/article/1496625
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            # 加载多个本地预训练模型
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
                # 循环优化过程，phase_num为一个全局配置，总循环优化阶段数
                for k in range(phase_num):
                    # iter_num设置为当前循环优化阶段的迭代数，phase_step为一个全局配置，各循环优化阶段的迭代数
                    iter_num = phase_step[k]
                    # curr_iternum目前推测为当前循环优化阶段的当前迭代控制变量，即curr_iternum == iter_num时退出当前循环优化阶段 \
                    #   判断逻辑在stop()中实现
                    # 循环优化的其中一个关键点，每个循环优化阶段初始迭代，循环优化梯度g重新施加在初始图像上 \
                    #   该关键点由此处每次循环时，x_input均固定设置为images来保证。
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

"""
Tensorflow读取并使用预训练模型：以inception_v3为例
https://blog.csdn.net/AManFromEarth/article/details/79155926
"""