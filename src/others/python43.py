import numpy as np

a=np.arange(9).reshape(3,3)
print(a)

# 水平分割  按列分割
print("水平分割")
# 将矩阵a划分为3个矩阵
b=np.hsplit(a,3)
print(b)

"""
# 相当于split函数中axis=1
b1=np.split(a,3,axis=1)
print(b1)
"""

b21 = np.hsplit(a, np.array([2, 6]))
print(b21)


"""
# 垂直分割
print("垂直分割")
c=np.vsplit(a,3)
print(c)
print("----------------\n")
c1=np.split(a,3,axis=0)
print(c1)
print("-----------------------------------\n")

# 深度分割
print("深度分割")
d=np.arange(27).reshape(3,3,3)
d1=np.dsplit(d,3)
print(d1)
print("-----------------------------------\n")
"""

array = np.array([0, 0])
for i in range(10):
    array = np.vstack((array, [i + 1, i + 1]))
print(array)

rand_arr = np.arange(array.shape[0])

np.random.shuffle(rand_arr)
print(array[rand_arr[0:5]])


np.random.shuffle(rand_arr)
print(array[rand_arr[0:5]])

