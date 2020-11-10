import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from scipy import interp


def roc_mean(fpr, tpr):
    n = len(fpr)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n
    return all_fpr, mean_tpr


fpr = np.array([[1, 5, 10], [2, 4, 8], [2, 6, 9]], dtype=float)
tpr = np.array([[2, 5, 9], [3, 6, 9], [1, 4, 10]], dtype=float)

fpr, tpr = roc_mean(fpr, tpr)
auc = auc(fpr, tpr)
print(auc)
plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

print(fpr, tpr)
