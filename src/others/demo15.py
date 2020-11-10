import numpy as np

""" numpy 学习 """

# 添加一行数据到当前np数组中
# train_data(97278, 8)
# train_label(97278, )
# test_data(283913, 8)
# test_label(283913, )
# train_size 97278

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.ones(3)
c = np.array([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
print(a)
print(b)
print(c)

c = np.insert(a, 0, values=b, axis=1)
print("a0", c)

c = np.insert(a, 3, values=b, axis=1)
print("a3", c)

# 添加一列数据到list中， list转为numpy
list = []
for i in range(90820):
    row = [1, 2, 3, 4, 5, 6, 7, 8]
    list.append(row)

data = np.array(list)

print(data.shape)

# 添加一列数据到list中， list转为numpy
list = []
for i in range(90820):
    row = 1
    list.append(row)

label = np.array(list)

print(label.shape)

# 现在添加的数据类型改变了， data (128, 8)
list = []
for j in range(500):
    data = []
    for i in range(128):
        row = [1, 2, 3, 4, 5, 6, 7, 8]
        data.append(row)
    data = np.array(data)
    print("data", data.shape)
    # 此时data.shape = (128, 8)， 如何把data添加到list中
    for k in range(len(data)):
        list.append(data[k])

list = np.array(list)
print(list.shape)


# 现在添加的数据类型改变了， label (128, 8)
list = []
for j in range(500):
    label = []
    for i in range(128):
        row = 1
        label.append(row)
    label = np.array(label)
    print("label", label.shape)
    # 此时label.shape = (128，) 如何把label添加到list中
    for k in range(len(label)):
        list.append(label[k])

list = np.array(list)
print(list.shape)
