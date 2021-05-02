from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


train_feat = np.array([[1, 5, 20],
                       [2, 7, 30],
                       [3, 21, 70],
                       [4, 30, 60],
                       ])
train_label = np.array([[0], [0], [1], [1]])

test_feat = np.array([[5, 25, 65]])
test_label = np.array([[1]])
print(train_feat.shape, train_label.shape, test_feat.shape, test_label.shape)

gbdt = GradientBoostingClassifier()
gbdt.fit(train_feat,train_label)
pred = gbdt.predict(test_feat)

total_err = 0
for i in range(pred.shape[0]):
    print(pred[i], test_label[i])
    err = (pred[i] - test_label[i]) / test_label[i]
    total_err += err * err
print(total_err / pred.shape[0])