import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score


def plot_roc(labels, predict_prob):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()


y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
scores = np.array([1, 0, 0.35, 0.8, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,0.33,0.44])
y_one = np.array([0, 0])
scores_one = np.array([1, 0])

# 单类数据无法绘制 roc 曲线
# plot_roc(y, scores)
plot_roc(y_one, scores_one)

# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
# print(fpr, tpr)
# plt.plot(fpr, tpr, marker='o')
#
# plt.show()
#
# from sklearn.metrics import auc
#
# AUC = auc(fpr, tpr)
