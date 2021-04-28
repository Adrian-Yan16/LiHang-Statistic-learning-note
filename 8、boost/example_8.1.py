import numpy as np
from sklearn.ensemble import AdaBoostClassifier

# 加载训练数据
X = np.array([[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2],
              [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]])

y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])


# sklearn 实例
adaboost = AdaBoostClassifier()
adaboost.fit(X,y)





















