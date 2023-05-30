# 公开的一种MIM实现，来自https://github.com/cleverhans-lab/cleverhans，用作参考

"""CA2基础班TF2版本 核心框架"""

import numpy as np
import tensorflow as tf

from ref.utils import optimize_linear, compute_gradient
from ref.utils import clip_eta

from TF2.conf_basic import *


def ca2_basic_tf2(
        model_fn,
        x,
        eps=max_epsilon,
        eps_iter=max_epsilon/255.0,
        nb_iter=iteration_num,
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

    pass


def loss_fn(labels, logits):
    """
    Added softmax cross entropy loss for MIM as in the original MI-FGSM paper.
    """

    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name=None)
