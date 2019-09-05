import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(roc_auc_score(y_true, y_scores))


y = np.array([1, 1, 2, 2])   #实际值
scores = np.array([0.1, 0.4, 0.35, 0.8])  #预测值
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)  #pos_label=2，表示值为2的实际值为正样本
print(fpr)
print(tpr)
print(thresholds)