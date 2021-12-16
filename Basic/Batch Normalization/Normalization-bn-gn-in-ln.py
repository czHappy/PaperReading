# coding;utf8
import torch
from torch import nn

# batch normalization
def test_BN():
    # track_running_stats=False，求当前 batch 真实平均值和标准差，
    # s
    # affine=False, 只做归一化，不乘以 gamma 加 beta（通过训练才能确定）
    # num_features 为 feature map 的 channel 数目
    # eps 设为 0，让官方代码和我们自己的代码结果尽量接近

    bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)

    x = torch.randn(4, 3, 2, 2)*100 #x.shape:[4,3,2,2]
    official_bn = bn(x)
    # print("x=",x)
    # 把 channel 维度单独提出来，而把其它需要求均值和标准差的维度融合到一起

    # x.permute(1, 0, 2, 3).shape: [c,n,h,w]
    # x.permute(1, 0, 2, 3).contiguous(): [c,n,h,w]
    # x.permute(1, 0, 2, 3).contiguous().view(3, -1): [c, n x h x w]

    # x1 = x.permute(1, 0, 2, 3).contiguous().view(3, -1)
    #  transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，
    #  而view操作要求tensor的内存连续存储，所以需要contiguous来返回一个contiguous copy
    x1 = x.transpose(0,1).contiguous().view(3,-1) # [c, n x h x w]
    # print("x1 = ",x1)
    # x1.mean(dim=1).shape: [3]
    mu = x1.mean(dim=1).view(1, 3, 1, 1) # [c, n x h x w] 对每个channel的所有元素求平均值=>[c,1]=>[1,c,1,1]

    # unbiased=False, 求方差时不做无偏估计（除以 N-1 而不是 N），和原始论文一致 unbiased = False
    # x1.std(dim=1).shape: [3]
    std = x1.std(dim=1, unbiased=False).view(1, 3, 1, 1) # [c, n x h x w] 对每个channel的所有元素求标准差 =>[c,1]=>[1,c,1,1]
    my_bn = (x - mu)/std # ([n,c,h,w] - [1,c,1,1]) / [1,c,1,1]  => [n,c,h,w]

    diff = abs(official_bn - my_bn).sum()
    # print(my_bn)

    print('diff={}'.format(diff))


# layer normalization
def test_LN():
    n ,c, h, w = 4,3,2,2
    x = torch.randn(n, c, h, w)*10000 #x.shape:[4,3,2,2]
    # normalization_shape 相当于告诉程序这本书有多少页，每页多少行多少列
    # eps=0 排除干扰
    # elementwise_affine=False 不作映射
    # 这里的映射和 BN 以及下文的 IN 有区别，它是 elementwise 的 affine，
    # 即 gamma 和 beta 不是 channel 维的向量，而是维度等于 normalized_shape 的矩阵
    ln = nn.LayerNorm(normalized_shape=[c, h, w], eps=0, elementwise_affine=False)

    official_ln = ln(x)

    # 把 N 维度单独提出来，而把其它需要求均值和标准差的维度融合到一起
    x1 = x.contiguous().view(n, -1)

    # x1.mean(dim=1).shape: [n]
    mu = x1.mean(dim=1).view(n, 1, 1, 1)

    # unbiased=False, 求方差时不做无偏估计（除以 N-1 而不是 N），和原始论文一致 unbiased = False
    std = x1.std(dim=1, unbiased=False).view(n, 1, 1, 1)
    my_ln = (x - mu) / std

    diff = abs(official_ln - my_ln).sum()
    # print(my_ln)

    print('diff={}'.format(diff))


# Instance Normalization
def test_IN():
    n, c, h, w = 4, 3, 2, 2
    x = torch.rand(n, c, h, w) * 10000
    In = nn.InstanceNorm2d(num_features=c, eps=0, affine=False, track_running_stats=False)

    offcial_in = In(x)

    x1 = x.view(n*c, -1)
    mu = x1.mean(dim=1).view(n, c, 1, 1)
    std = x1.std(dim=1, unbiased=False).view(n, c, 1, 1)

    my_in = (x - mu) / std

    diff = abs(my_in - offcial_in).sum()
    print('diff={}'.format(diff))

def test_GN():
    n, c, h, w = 10, 20, 5, 5
    g = 4
    x = torch.rand(n, c, h, w) * 1
    # 分成 4 个 group
    gn = nn.GroupNorm(num_groups=g, num_channels=c, eps=0, affine=False)
    official_gn = gn(x)

    # 把同一个group的元素融合到一起
    # 分成 g 个 group

    x1 = x.view(n, g, -1)
    mu = x1.mean(dim=-1).reshape(n, g, -1)
    std = x1.std(dim=-1).reshape(n, g, -1)

    x1_norm = (x1 - mu) / std
    my_gn = x1_norm.reshape(n, c, h, w)

    diff = abs(my_gn - official_gn).sum()

    print('diff={}'.format(diff))


test_BN()
test_LN()
test_IN()
test_GN()