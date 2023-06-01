import numpy as np


def gkern(kernlen=21, nsig=3):
    """
    Returns a 2D Gaussian kernel array.
    这是一个副本，来自于原TF1版本中utils.py
    """

    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)

    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel


if __name__ == "__main__":
    stack_kernel_9 = gkern(9, 3)
    print(stack_kernel_9.shape)
