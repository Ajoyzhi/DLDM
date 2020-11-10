import torch
import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch


y1 = [0, 0, 0, 0, 0]
y2 = [1, 1, 1, 1, 1]

print(torch.tensor(y1))
print(torch.tensor(y2))

print(torch.Tensor(y1))
print(torch.Tensor(y2))


y1 = np.ndarray(y1)
y2 = np.ndarray(y2)

print(torch.from_numpy(y1))
print(torch.from_numpy(y2))

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(b)

c = np.zeros(10)
c[1] = 10
print(torch.Tensor(c))

x_mat = np.zeros((10, 8))



