import torch

a =  torch.Tensor([[1,2,3],
                  [2,3,4],
                  [3,4,5]])
b = torch.Tensor([[1,1,1]])
# 对应元素相乘
c = torch.sum((a - b) ** 2, dim=1)

print(c)