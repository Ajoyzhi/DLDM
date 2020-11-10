from __future__ import print_function
from torch.autograd import Variable

import logging
import random
import numpy as np
import torch

'''
    torch 教程 -- tensor 基本操作
'''
print("###################  torch 教程 -- tensor 基本操作  ####################")
x = torch.tensor([5.5, 3])
print(x.size())

x = torch.from_numpy(np.ones([3, 3, 3]))
print(x.size())


x = torch.Tensor(5, 3)  # 构造一个未初始化的5*3的矩阵
x = torch.rand(5, 3)  # 构造一个随机初始化的矩阵

print(x)
print(x.size())


y = torch.rand(5, 3)
print(y)

# 相同结构的tensor相加
print(x+y)
print(torch.add(x, y))

# 另外输出tensor也有两种写法
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
y.add_(x)
print(result)

# 特别的是, 任何改变tensor的操作都会在方法名后面有一个下划线 "_"

# tensor 支持python切片操作
print(x[:, 1])

'''
    torch 教程 -- numpy
'''
print("###################  torch 教程 -- numpy  ####################")
# tensor和numpy共用存储空间，同时修改
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(b)  # 修改a，b也会修改

# 将numpy数组转换为torch的tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# 当gpu可用时，可以切换到gpu运算
# 使用cuda函数来讲tensor移动到gpu上
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y

'''
    torch 教程 -- 反向传播
        autograd 为tensor提供了自动求导，
        requires_grad = True：追踪tensor的操作，便于求导
        .backward()：自动计算梯度，并保存到每个tensor的.grad属性上
        .data：返回对象所包裹的tensor
'''

print("###################  torch 教程 -- 反向传播  ####################")

# 创建一个Variable，包裹一个2*2的张量，将需要计算题都属性设置True
x = Variable(torch.ones(2, 2), requires_grad = True)
y = x + 2
y.grad_fn  # 每个Variable都有一个creator（创造者节点）

z = torch.mean(y * y)
print("z", z.data)


# 反向传播
z.backward()
print(z.grad)
print(y.grad)
print(x.grad)

# 例子
s = Variable(torch.FloatTensor([[0.01, 0.02]]), requires_grad = True)
x = Variable(torch.ones(2, 2), requires_grad = True)
print(s.size())
print(x.size())
for i in range(10):
    s = s.mm(x)
z = torch.mean(s)
z.backward()
print(x.grad)
print(s.grad)







