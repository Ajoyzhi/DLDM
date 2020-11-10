import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score

# y = np.array([2, 2, 2, 2])
# scores = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
# print(fpr, tpr, thresholds)

y_pred = [0, 1, 1, 1]
y_true = [0, 0, 0, 1]
auc = accuracy_score(y_true, y_pred)
print(auc)

auc = roc_auc_score(y_true, y_pred)
print(auc)
