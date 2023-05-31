"""CA2基础班TF2版本 核心框架"""

# import numpy as np
import tensorflow as tf

from TF2.conf_basic import *


def ca2_basic_tf2(
        model_fn,
        x,
        # eps=12.0,                             # 直接获取全局配置
        # eps_iter=2.0 * 12.0 / 255.0,          # 根据全局配置进行计算，不支持额外自定义
        # nb_iter=16,                           # 直接使用全局循环优化配置
        norm=np.inf,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        # decay_factor=momentum_decay_factor,   # 直接使用全局配置
        sanity_checks=True,
):
    """
    CA2基础版本的TF2版本实现。
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    # :param eps: (optional float) maximum distortion of adversarial example
    #           compared to original input
    # :param eps_iter: (optional float) step size for each attack iteration
    # :param nb_iter: (optional int) Number of attack iterations.
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
        raise ValueError("norm当前仅支持设置为: np.inf")

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
        adv_x, cyclical_momentum = graph(x, iter_num, cyclical_momentum, model_fn, loss_fn, y, targeted)

    if sanity_checks:
        assert np.all(asserts)

    return adv_x


def compute_gradient(model_fn, loss_fn, x, y, targeted):
    """
    循环优化的某一阶段，由原样本x初始优化，迭代iter_num，循环优化动量继承自上一循环优化阶段
    :param model_fn:
    :param loss_fn:
    :param x: 输入样本张量
    :param y: 输入样本对于的真实标签
    :param targeted: bool，True：targeted攻击，False：non-targeted攻击
    :return: 当前迭代计算得到的梯度值
    """

    # 随机采样得到偏移向量
    vector = tf.random_normal(shape=x.shape)  # 需要找一下TF2版本的API

    # sample_variance为偏移距离（文章中的w），可以理解为图像每个像素的偏移量
    # 通过sign函数符号化偏移向量，可以控制每像素偏移值为 -w、0、w
    x_nes = x + sample_variance * tf.sign(vector)

    # 根据配置的本地白盒模型网络类型，返回相应模型对于x_nes的输出
    logits = get_logits(x_nes, model_fn)


    pass


def graph(x, iter_num, cyclical_momentum, model_fn, loss_fn, y, targeted):
    """
    循环优化的某一阶段，由原样本x初始优化，迭代iter_num，循环优化动量继承自上一循环优化阶段
    :param x: 循环优化初始样本，由原样本x初始优化
    :param iter_num: 当前循环优化阶段迭代次数
    :param cyclical_momentum: 上一阶段的循环优化动量
    :param model_fn:
    :param loss_fn:
    :param y:
    :param targeted:
    :return: 当前循环优化阶段结束是的对抗样本adv_x，以及当前阶段结束时的循环优化动量cyclical_momentum
    """

    # 设置最大扰动约束：2*12/255 ， 文章中不是提到是 12/255吗
    #   这个疑问黄博已解答，见《几个代码疑问.docx》之问题1/3
    eps = 2.0 * max_epsilon / 255.0

    # 设置当前阶段的步长，即文章中的 α = e/N(t)
    alpha = eps / float(iter_num)

    adv_x = x

    # 根据循环优化策略，动量初始化为上一阶段循环优化动量 * 学习权重
    momentum = cyclical_momentum * momentum_learn_factor

    for k in range(iter_num):
        # 进入第k次迭代的梯度计算，样本为上一次迭代输出的对抗样本；
        grad = compute_gradient(model_fn, loss_fn, adv_x, y, targeted)
        # Normalize current gradient and add it to the accumulated gradient
        # tf.math.reduce_mean: https://www.w3cschool.cn/tensorflow_python/tensorflow_python-hckq2htb.html
        red_ind = list(range(1, len(grad.shape)))
        avoid_zero_div = tf.cast(1e-12, grad.dtype)
        grad = grad / tf.math.maximum(
            avoid_zero_div,
            tf.math.reduce_mean(tf.math.abs(grad), red_ind, keepdims=True),
        )

        momentum = grad + momentum_decay_factor * momentum

        # 更新当前循环优化阶段当前迭代的样本x
        # 循环优化的另一个关键点：在每个循环优化阶段的初始迭代中，计算得到的循环优化梯度g重新施加在初始图像张量x上 \
        #   这一点在该函数中未体现，该函数只呈现单个迭代逻辑，总是将当前迭代计算的g施加在入参样本x上，至于循环优化关键点由main()中循环优化阶段 \
        #   的for循环中特别注意每个循环传入的入参x必须为原始图像张量
        adv_x = adv_x + alpha * tf.sign(momentum)

    cyclical_momentum = momentum
    return adv_x, cyclical_momentum


def loss_fn(labels, logits):
    """
    Added softmax cross entropy loss for MIM as in the original MI-FGSM paper.
    """

    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name=None)
