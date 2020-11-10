"""
    二分类问题，绘制roc曲线
    注意： 标签只有0,1
"""
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from utils.math_utils import init_list, add_list, print_list, load_scores_labels, save_scores_labels


def plot_roc(labels, pred, roc_auc):
    """ 绘制 roc 曲线 """
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(labels, pred)
    print("false_positive_rate", false_positive_rate)
    print("true_positive_rate", true_positive_rate)

    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()


labels = [1, 0, 0, 0, 1, 1, 1, 1, 1, 0]
pred = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]


