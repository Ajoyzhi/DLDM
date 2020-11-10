import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


a = np.array([0, 2.2323, 3.232, 4.23])
b = np.array([0, 2.1, 3.123123, 4.13123])
a += b
print(a.shape)
print(a)
a = a / 2
print(a)

# -*- coding: utf-8 -*-

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

salary = [2500, 3300, 2700, 5600, 6700, 5400, 3100, 3500, 7600, 7800,
          8700, 9800, 10400]

group = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]


plt.hist(salary, group, histtype='bar', rwidth=0.8)

plt.legend()

plt.xlabel('salary-group')
plt.ylabel('salary')

plt.title(u'测试例子——直方图', FontProperties=font)

plt.show()
