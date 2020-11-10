import numpy as np


""" numpy 保存为 npy 格式文件 """

# 不仅是保存为txt,excel 等，也可以保存为.npy文件，可以保存为相应的数组格式
data = np.array([[1, 2, 3], [4, 5, 6]])
np.save('data_test.npy', data)
new = np.load('data_test.npy')
print(new)
print(new)
