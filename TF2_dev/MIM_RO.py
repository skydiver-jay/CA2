# 先搞一个RO版本的MIM，能够正常运行，再加入SA
#   引入SA，总迭代次数需要分为多个循环优化阶段，阶段配置引用conf_basic.py中的配置
#   另外，还需要新增学习权重参数-momentum_learn_factor，同样引用conf_basic.py中的配置

"""The MomentumIterativeMethod attack."""

import numpy as np
import tensorflow as tf

# from cleverhans.tf2.utils import optimize_linear, compute_gradient
# from cleverhans.tf2.utils import clip_eta
from ref.utils import optimize_linear, compute_gradient
from ref.utils import clip_eta

from conf_basic import *


def momentum_iterative_method(
        model_fn,
        x,
        # eps=0.3,  # 改为使用全局配置
        # eps_iter=0.06,  # 改为使用全局配置
        # nb_iter=10,  # 改为使用全局配置
        norm=np.inf,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        # decay_factor=1.0,  # 改为使用全局配置
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
    :param eps: (optional float) maximum distortion of adversarial example
              compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
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
    :param decay_factor: (optional) Decay factor for the momentum term.
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
            grad = compute_gradient(model_fn, loss_fn, adv_x, y, targeted)

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


def loss_fn(labels, logits):
    """
    Added softmax cross entropy loss for MIM as in the original MI-FGSM paper.
    """

    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name=None)
