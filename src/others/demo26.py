import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator

# 遍历dict

dict = {'a': None, 'b': 2, 'c': 3, }
for i in dict:
    print(dict[i])

# 生成柱状图
# 显示高度

x_major_locator = MultipleLocator(1)
# 把x轴的刻度间隔设置为1，并存在变量里
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
name_list = ['svdd', 'lstm_svdd', 'join']
num_list = [0.9820548203171366, 0.9938785275247235, 0.9960687051467705]
plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list, width=0.1)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
