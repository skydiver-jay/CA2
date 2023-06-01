"""CA2基础班TF2版本 配置文件"""

import numpy as np
from TF2.utils_tf2 import *
import random

"""
此配置文件仅包括CA2中RO和SA相关配置。
本地模型和图像相关配置使用单独配置文件。
"""

# 原图最大扰动范围
max_epsilon = 12.0

# 动量衰减系数/动量衰减权重
momentum_decay_factor = 1.0

# sample_num: 属于CA2框架“偏移增强”特性的一个参数。 文章中的k，生成偏移增强样本的数量，也即是随机采样的数量
# sample_variance: 为偏移距离，即文章中的ω，文章中实验设置为0.05，此处设置为0.1，待确认哪个是默认推荐配置
#   这个疑问黄博已解答，见《几个代码疑问.docx》之问题1
sample_num = 4
sample_variance = 0.1

# 循环优化策略配置：
#   如下配置表示，将迭代总数为16的过程，分为3个阶段进行循环优化，各个阶段的迭代次数为4、4、8，此为文章中实验推荐配置
#   momentum_learn_factor为学习权重，文章中实验推荐配置为1.0
phase_step = [4, 4, 8]
phase_num = 3
iteration_num = 16
momentum_learn_factor = 1.0

# 配置卷积核尺寸分别为9和11的两个高斯卷积矩阵
#   文章中结论：选择卷积核尺寸集合为 {1,9,11}，默认情况下建议使用此配置；当卷积核尺寸为1时，实则为原模型不变，这个在后续全局梯度计算相关代码中有体现
list_stack_kernel_size = [1, 9, 11]
# 如果需要自定义，只需要变更上一行数组中的奇数配置即可，以下3行代码无需变动
stack_kernel = gkern(list_stack_kernel_size[0], 1)
stack_kernel_9 = gkern(list_stack_kernel_size[1], 3)
stack_kernel_11 = gkern(list_stack_kernel_size[2], 3)

# 参数合法性检查
if phase_num != len(phase_step):
    raise ValueError("循环优化阶段数配置错误: 总阶段数与分阶段配置数组长度不一致")

n = 0
for i in phase_step:
    n += i
if n != iteration_num:
    raise ValueError("循环优化阶段数配置错误: 总迭代数与分阶段配置数组总和不一致")