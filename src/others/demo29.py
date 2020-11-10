import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, roc_curve, auc

""" 计算roc，mcc """
# y_pred = np.array([0, 1, 1, 1, 1])
# y_test = np.array([0, 1, 1, 1, 1])
# mcc = matthews_corrcoef(y_test, y_pred)
# print(mcc)
#
# y_test = [0, 1, 1, 1, 1]
# accuracy_score(y_test, y_pred)
#
# for j in range(2,5):
#     print(j)

# y_pred = np.array([0.1, 0.1, 0.9, 0.9])
# y_test = np.array([0, 1, 1, 1])
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
# print(thresholds)
# print(thresholds.shape)


import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def roc(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = find_optimal_cutoff(tpr=tpr, fpr=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point


def find_optimal_cutoff(tpr, fpr, threshold):
    """ 寻找最优阀值 -- 阿登指数 """
    y = tpr - fpr
    youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[youden_index]
    point = [fpr[youden_index], tpr[youden_index]]
    return optimal_threshold, point


y = np.array([0., 0., 1., 1.])
pred = np.array([0.1232342, 0.4234234, 0.35234234, 0.823423423])  # 0.8 0 0 0 1 < >=
fpr, tpr, roc_auc, optimal_th, optimal_point = roc(y, pred)
# optimal_th = np.array(optimal_th)
pred_labels = np.array(list(map(lambda x: 0.0 if x < optimal_th else 1.0, pred)))

print(pred_labels)
plt.figure(1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
plt.title("ROC-AUC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()



