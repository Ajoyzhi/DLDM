import numpy as np

'''
维度变换

'''

x = np.zeros([100, 28, 128])
print(x.shape)

y = x[:, -1, :]
print(y.shape)


y = x[-1, :, :]
print(y.shape)


x = np.zeros([150, 3])
print(x.shape)
y = x.view(-1, 1, 3)
print(y.shape)




