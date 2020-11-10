from __future__ import print_function
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    线性层 Linear
'''
# [128, 20] --> [128, 30]
m = nn.Linear(20, 30)
input = Variable(torch.randn(128, 20))
output = m(input)
print(output.size())

'''
    卷积层 Conv2d
'''
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
# m = nn.Conv2d(16, 33, 3)
m = nn.Conv2d(1, 6, 3)
# input = Variable(torch.randn(20, 16, 50, 100))
input = Variable(torch.randn(28,1,28,28))
output = m(input)
print(output.size())

'''
    池化层 F.max_pool2d / nn.MaxPool1d
'''

m = nn.MaxPool1d(3, stride=2)
input = Variable(torch.randn(20, 16, 50))
output = m(input)
print(output.size())


input = Variable(torch.randn(1, 16, 13, 13))
output = F.max_pool2d(input, (2, 2))
print(output.size())

