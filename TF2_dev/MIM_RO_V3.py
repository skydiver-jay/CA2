# 基于MIM-RO-V2版本，尝试加入虚拟集成VME算子
#   VME策略中需要配置的卷积核尺寸及各尺寸卷积矩阵在全局配置中初始化


"""采用循环优化RO策略，并且集成D2A & VME策略的MIM"""

import numpy as np
import tensorflow as tf

# ref.utils来自于cleverhans项目源码
from ref.utils import optimize_linear, compute_gradient
from ref.utils import clip_eta

from conf_basic import *


def momentum_iterative_method(
        model_fn,
        x,
        norm=np.inf,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        sanity_checks=True,
):
    """
    Tensorflow 2.0 implementation of Momentum Iterative Method (Dong et al. 2017).
    This method won the first places in NIPS 2017 Non-targeted Adversarial Attacks
    and Targeted Adversarial Attacks. The original paper used hard labels
    for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param norm: (optional) Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """

    # Check if order of the norm is acceptable given current implementation
    if norm not in [np.inf]:
        raise ValueError("norm当前仅支持 np.inf.")

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    # 此处处理target/non-target攻击区别，如果是target攻击，y由外部传入；如果是non-target攻击，y由本地模型预测得到
    # y是后续计算梯度的必须参数
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        # TensorFlow 返回张量的最大值索引API，https://www.w3cschool.cn/tensorflow_python/tensorflow_python-lbm22c8b.html
        y = tf.argmax(model_fn(x), 1)

    # Initialize loop variables
    momentum = tf.zeros_like(x)

    for i in range(phase_num):
        # 进入第i个循环优化阶段
        print("---- 进入第%d个循环优化阶段, 当前阶段迭代总数为%d ----" % (i+1, phase_step[i]))

        # 每个循环优化阶段开始时，样本均初始化为原始样本（即重新出发）
        print("每个循环优化阶段开始时，样本均初始化为原始样本")
        adv_x = x
        # 根据循环优化策略，动量初始化为上一阶段循环优化动量 * 学习权重
        print("每个循环优化阶段开始时，动量初始化为上一阶段循环优化动量 * 学习权重")
        momentum = momentum * momentum_learn_factor

        # 进入第i个循环优化阶段的第j次迭代
        for j in range(phase_step[i]):
            print("---- 进入第%d个循环优化阶段的第%d次迭代 ----" % (i+1, j+1))
            # 计算梯度
            #   根据‘认识CA2.md’中的分析，SA策略中将要提取的算子，基本都是在计算损失函数前，对输入样本的变换（除TIM策略外）
            #   V2版本加入D2A策略，D2A策略会随机采样sample_num个偏移方向，每个方向经过偏移sample_variance后获得，sample_num个偏移增强样本
            #   使用sample_num个偏移增强样本计算梯度值，最终综合梯度等于累积梯度/sample_num
            print("开始计算D2A策略综合梯度")
            grad = tf.zeros_like(x)
            for k in range(sample_num):
                print("---- 进入第%d个循环优化阶段的第%d次迭代，第%d个偏移样本梯度计算 ----" % (i + 1, j + 1, k+1))
                x_nes = transformation_d2a(adv_x)
                # VME等虚拟模型方向类的策略，集成点在logits计算后，需要改造原本的compute_gradient()
                grad = grad + compute_gradient_ca2(model_fn, loss_fn, x_nes, y, targeted)
            grad = grad / sample_num

            # 计算累积梯度：这一部分可以抽象出来作为一个独立算子，替换该算子，即可集成不同的优化算法，如从MI -> NI
            # tf.math.reduce_mean: https://www.w3cschool.cn/tensorflow_python/tensorflow_python-hckq2htb.html
            red_ind = list(range(1, len(grad.shape)))
            avoid_zero_div = tf.cast(1e-12, grad.dtype)
            grad = grad / tf.math.maximum(
                avoid_zero_div,
                tf.math.reduce_mean(tf.math.abs(grad), red_ind, keepdims=True),
            )
            momentum = momentum_decay_factor * momentum + grad
            # 计算累积梯度 end

            # 此处的实现与原文伪代码不一致
            #   根据MIM原文伪代码 optimal_perturbation 应等于 步长 * sign(momentum)，CA2中的实现与原文伪代码一致
            #   但此处的optimize_linear()，其中针对不同范数，对momentum的应用不一样
            #       当norm==tf.inf时，optimize_linear()的行为与原文一致，所以使用optimize_linear()是只使用norm==tf.inf场景
            #       当前迭代的步长为eps_iter：最大扰动范围 / 第i个循环优化阶段的迭代总数
            eps_iter = max_epsilon / phase_step[i]
            optimal_perturbation = optimize_linear(momentum, eps_iter, norm)
            # 更新对样样本
            adv_x = adv_x + optimal_perturbation
            # 这一步用于确保每一个像素的扰动范围都在定义的最大扰动max_epsilon范围内，功能类似tf.clip_by_value
            adv_x = x + clip_eta(adv_x - x, norm, max_epsilon)

            # 这一步控制，每一轮迭代更新后，样本x的变动不超出定义的最大有效范围，可参考CA2.py-main()中的x_max和x_min
            if clip_min is not None and clip_max is not None:
                adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def transformation_d2a(x):
    """
    采用D2A策略对输入样本进行’增强‘变换
    :param x: 输入样本
    :return: 根据D2A变换后的增强样本
    """
    # CA2.py中由于使用inception模型，样本均归一化到(-1,1). sample_variance的配置也作用在归一化后的样本上，最优配置为0.1。
    #   实则文中在(0,1)语境下表述，最优配置为0.05
    # 本代码中本地使用ResNet模型，样本为(0,255)，因此在应用D2A策略相关配置时，将样本先归一化到(0,1)，计算完在还原至(0,255)
    x = x / 255.0

    # 随机采样得到偏移向量
    vector = tf.random.normal(shape=x.shape)

    # sample_variance为偏移距离（文章中的ω），可以理解为图像每个像素的偏移量
    # 通过sign函数符号化偏移向量，可以控制每像素偏移值为 -ω、0、ω
    x_nes = x + sample_variance * tf.sign(vector)

    x_nes = x_nes * 255.0
    return x_nes


@tf.function
def compute_gradient_ca2(model_fn, loss_fn, x_nes, y, targeted):
    """
    Computes the gradient of the loss with respect to the input tensor.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param loss_fn: loss function that takes (labels, logits) as arguments and returns loss.
    :param x_nes: 输入样本，实施样本增强后的样本
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :return: A tensor containing the gradient of the loss with respect to the input tensor.
    """

    # tf.GradientTape(), 参考https://blog.csdn.net/guanxs/article/details/102471843
    with tf.GradientTape() as g:
        g.watch(x_nes)

        stack_kernel_num = len(list_stack_kernel_size)
        base_logits = model_fn(x_nes)
        logits = tf.zeros_like(base_logits)
        for i in range(stack_kernel_num):
            # 如果卷积核尺寸为1*1，则相当于不变，则无需真的进行卷积计算，直接累加原logits
            if list_stack_kernel_size[i] == 1:
                logits = logits + base_logits
            # 如卷积核尺寸不为1*1，则卷积计算后，再计算logits，并累加
            else:
                x_conv = tf.nn.depthwise_conv2d(x_nes, stack_kernel_list[i], strides=[1, 1, 1, 1], padding='SAME')
                logits = logits + model_fn(x_conv)
        # 多个虚拟模型输出的均值，用于后续计算当前偏移增强样本的损失函数
        logits = logits / stack_kernel_num

        # Compute loss
        # 若要集成VME等虚拟模型方向类的策略，此处logits的输入为多个虚拟模型的综合logits，而不再是针对单个输入样本的logits
        #   改造原本compute_gradient() -> compute_gradient_ca2()
        loss = loss_fn(labels=y, logits=logits)
        if (
            targeted
        ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x_nes)
    return grad


def loss_fn(labels, logits):
    """
    Added softmax cross entropy loss for MIM as in the original MI-FGSM paper.
    """

    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name=None)

