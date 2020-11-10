# Credit: Josh Hemann

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

n_groups = 2
aucs = [0.9820548203171366, 0.9938785275247235, 0.9960687051467705]  # 准确率
times = (np.array([2.7140002250671387, 8.94599962234497, 6.049999713897705]) / 8.94599962234497).tolist()  # 时间

ssvdd = (aucs[0], times[0])

lstm_svdd = (aucs[1], times[1])

join = (aucs[2], times[2])

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.05

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, ssvdd, bar_width,
                alpha=opacity, color='r', error_kw=error_config,
                label='svdd')

rects2 = ax.bar(index + bar_width, lstm_svdd, bar_width,
                alpha=opacity, color='b'
                , error_kw=error_config,
                label='lstm+svdd')

rects3 = ax.bar(index + bar_width * 2, join, bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='lstm+svdd+join')

ax.set_xlabel('types')
ax.set_title('roc_self compare result')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(('auc', 'time'))
ax.legend()

fig.tight_layout()
plt.show()
