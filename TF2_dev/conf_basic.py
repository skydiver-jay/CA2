"""CA2基础班TF2版本 配置文件"""

from utils_tf2 import *

"""
此配置文件仅包括CA2中RO和SA相关配置。
本地模型和图像相关配置使用单独配置文件。
"""

# 仅当不需要真随机场景时，将need_true_random配置为False
need_true_random = True
seed = 0
if not need_true_random:
    tf.random.set_seed(seed)

# 原图最大扰动范围，CA2文中实验设置为12.0，但在ImageNet数据集和ResNet模型上实验结果噪声过于明显。
# 本地实验显示，设置为3.0，获得的对抗样本噪声几乎不影响视觉效果。
max_epsilon = 3.0

# 动量衰减系数/动量衰减权重
momentum_decay_factor = 1.0


##########################           偏移增强D2A策略配置           ##########################
# sample_num: 属于CA2框架“偏移增强”策略的一个参数。 文章中的k，生成偏移增强样本的数量，也即是随机采样的数量
# sample_variance: 为偏移距离，即文章中的ω，文章中实验设置为0.05，此处设置为0.1
#   TF1版本CA2.py中使用0.1，是因为本地模型为inception，样本归一化到(-1,1)
#   文章中使用0.05，是因为在(0,1)语境下表述
#   此处采用0.05，那么在D2A策略实现代码中，样本的bounds将样本归一化到(0,1)后再进行计算
sample_num = 4
sample_variance = 0.05

##########################           循环优化RO策略配置           ##########################
#   如下配置表示，将迭代总数为16的过程，分为3个阶段进行循环优化，各个阶段的迭代次数为4、4、8，此为文章中实验推荐配置
#   momentum_learn_factor为学习权重，文章中实验推荐配置为1.0
phase_step = [4, 4, 8]
phase_num = 3
iteration_num = 16
momentum_learn_factor = 1.0

##########################           虚拟集成VME策略配置           ##########################
# 配置卷积核尺寸分别为1/9/11的高斯卷积矩阵
#   文章中结论：选择卷积核尺寸集合为 {1,9,11}，默认情况下建议使用此配置；当卷积核尺寸为1时，实则为原模型不变，这个在后续全局梯度计算相关代码中有体现
list_stack_kernel_size = [1, 9, 11]
# 如果需要自定义，只需要变更上一行数组中的奇数配置即可，以下3行代码无需变动
stack_kernel_1 = gkern(list_stack_kernel_size[0], 3)  # 实际上不起作用
stack_kernel_2 = gkern(list_stack_kernel_size[1], 3)
stack_kernel_3 = gkern(list_stack_kernel_size[2], 3)
stack_kernel_list = [stack_kernel_1, stack_kernel_2, stack_kernel_3]

# 参数合法性检查
if phase_num != len(phase_step):
    raise ValueError("循环优化阶段数配置错误: 总阶段数与分阶段配置数组长度不一致")

n = 0
for i in phase_step:
    n += i
if n != iteration_num:
    raise ValueError("循环优化阶段数配置错误: 总迭代数与分阶段配置数组总和不一致")


# 打印全局参数
def print_global_conf(config, desc):
    print("%s: %s" % (desc, str(config)))


print_global_conf(max_epsilon, "最大噪声范围")
print_global_conf(momentum_decay_factor, "动量衰减系数")

print_global_conf(phase_step, "RO 之 循环优化分阶段配置")
print_global_conf(momentum_learn_factor, "RO 之 循环优化学习权重")

print_global_conf(sample_num, "D2A 之 偏移增强样本个数")
print_global_conf(sample_variance, "D2A 之 偏移距离")

print_global_conf(list_stack_kernel_size, "VME 之 卷积核尺寸配置")

print_global_conf(tf.__version__, "TensorFlow版本")

if not need_true_random:
    print_global_conf(seed, "随机种子")
