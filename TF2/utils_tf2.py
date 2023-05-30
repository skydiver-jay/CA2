import numpy as np


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    """
        scipy包含各种专用于科学计算中常见问题的工具箱。其不同的子模块对应不同的应用，如插值、积分、优化、图像处理、统计、特殊函数等。 https://zhuanlan.zhihu.com/p/462806946
        scipy.stats: 统计和随机数
            scipy.stats.norm是一个分布对象：每个分布都表示为一个对象。norm 是正态分布(也就是高斯分布)，包含有 PDF、CDF 等等。

    """
    import scipy.stats as st

    """
        numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)  https://zhuanlan.zhihu.com/p/452436216
        功能: 生成一个指定大小，指定数据区间的均匀分布序列
        此处，均值为0，取值范围为[-nsig, nsig]，采样数据点个数为kernlen
    """
    x = np.linspace(-nsig, nsig, kernlen)

    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel