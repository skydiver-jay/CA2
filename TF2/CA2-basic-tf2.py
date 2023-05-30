"""CA2基础班TF2版本 核心框架"""

import numpy as np
import tensorflow as tf

from TF2.conf_basic import *


def ca2_basic_tf2(
        model_fn,
        x,
        eps=12.0,
        eps_iter=2.0*12.0/255.0,
        nb_iter=16,
        norm=np.inf,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        decay_factor=momentum_decay_factor,
        sanity_checks=True,
):
    """
    CA2基础版本的TF2版本实现。
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: (optional float) maximum distortion of adversarial example
              compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param norm: (预留) 当前只支持默认值: np.inf.
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
    :param decay_factor: (optional) Decay factor for the momentum term.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """

    if norm != np.inf:
        raise ValueError("norm当前仅支持设置为: np.inf.")

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
        y = tf.argmax(model_fn(x), 1)

    # Initialize loop variables
    cyclical_momentum = tf.zeros_like(x)

    adv_x = x

    # 循环优化过程，phase_num为一个全局配置，总循环优化阶段数
    for k in range(phase_num):
        # iter_num设置为当前循环优化阶段的迭代数，phase_step为一个全局配置，各循环优化阶段的迭代数
        iter_num = phase_step[k]
        # 进入第k阶段循环优化，起点x、阶段内迭代数iter_num、循环优化动量初始值cyclical_momentum
        adv_x, cyclical_momentum = graph(x, iter_num, cyclical_momentum)

    pass


def compute_gradient():
    pass


def graph(x, iter_num, cyclical_momentum):
    pass


def loss_fn(labels, logits):
    """
    Added softmax cross entropy loss for MIM as in the original MI-FGSM paper.
    """

    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name=None)
