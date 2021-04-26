import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# from sklearn.model_selection import train_test_split


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'labels'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(data.shape[0]):
        if data[i][-1] == 0:
            data[i][-1] = -1
    return data[:, :-1], data[:, -1]


X, Y = create_data()

fig, ax = plt.subplots(figsize=(9, 5))
pos = X[Y == 1]
neg = X[Y == -1]

ax.scatter(pos[:, 0],
           pos[:, 1],
           c='r',
           marker='o',
           label='1')
ax.scatter(neg[:, 0],
           neg[:, 1],
           c='b',
           marker='x',
           label='-1')

ax.legend()

# plt.show()


class SVM:
    def __init__(self, max_iters, kernal='linear', C=1.0, p=2):
        self.max_iters = max_iters
        self._kernal = kernal
        self.C = C
        self.p = p

    def init_args(self, features, labels):
        # 初始化参数
        self.m, self.n = features.shape
        self.x_train = features
        self.y_train = labels
        self.b = 0.0
        # 将 E 保存于列表中
        self.alpha = np.zeros(self.m)
        self.E = [self.cal_E(i) for i in range(self.m)]

    def KKT(self, idx):
        # 判断是否满足 KKT 条件
        y_g = self.y_train[idx] * self.cal_g(idx)
        if self.alpha[idx] == 0:
            return y_g >= 1
        elif 0 < self.alpha[idx] < self.C:
            return y_g == 1
        elif self.alpha[idx] == self.C:
            return y_g <= 1

    def cal_a1_a2(self, idxs):
        # 找到 a1 和 a2
        for i in idxs:
            if self.KKT(i):
                continue
            # 内层循环
            E1 = self.E[i]
            # alpha2 依赖于 |E1 - E2|，如果 E1 为正，选择 E2 最小的为 alpha2
            # 否则选择最大的
            if E1 >= 0:
                j = self.E.index(min(self.E))
            else:
                j = self.E.index(max(self.E))
            return i, j
        return None, None

    def init_alpha(self):
        # 外层循环找第一个变量,违反 KKT 条件最严重的样本点
        support_vec = np.argwhere(self.alpha < self.C)
        support_vec = np.array([i[0] for i in support_vec if self.alpha[i[0]] > 0])

        a1, a2 = self.cal_a1_a2(support_vec)
        if not a1:
            # 如果都满足 KKT 条件，遍历整个数据集
            non_satisfy = np.array([i for i in range(self.alpha.size) if i not in support_vec])
            a1, a2 = self.cal_a1_a2(non_satisfy)
        return a1, a2

    def cal_g(self, idx):
        # 预测值
        alpha_y = np.multiply(self.alpha, self.y_train)
        K = self.kernal(self.x_train, self.x_train[idx])
        return np.dot(alpha_y, K) + self.b

    def cal_E(self, idx):
        # 计算 E，即预测值与真实值的差
        return self.cal_g(idx) - self.y_train[idx]

    def kernal(self, x1, x2):
        # 核函数
        if self._kernal == 'linear':
            return np.dot(x1, x2)
        elif self._kernal == 'poly':
            return np.power((np.dot(x1, x2) + 1), 2)
        else:
            return 0

    def cut_along_condition(self, L, H, E1, E2, a1, a2):
        # 沿着约束方向剪辑后的 a2
        eta = self.kernal(self.x_train[a1], self.x_train[a1]) + \
              self.kernal(self.x_train[a2], self.x_train[a2]) - \
              self.kernal(self.x_train[a1], self.x_train[a1]) * 2

        if eta <= 0:
            return None

        a2_new_unc = self.alpha[a2] + self.y_train[a2] * (E1 - E2) / eta

        if a2_new_unc > H:
            return H
        elif a2_new_unc < L:
            return L
        return a2_new_unc

    def cal_b_new(self, E, a1, a2, a1_new, a2_new):
        # 计算 b 的值
        first_part1 = -E - self.y_train[a1] * self.kernal(self.x_train[a1], self.x_train[a1]) * (a1_new - self.alpha[a1])
        second_part1 = - self.y_train[a2] * self.kernal(self.x_train[a2], self.x_train[a1]) * (a2_new - self.alpha[a2]) + self.b

        return first_part1 + second_part1

    def fit(self, features, labels):
        # 初始化参数
        self.init_args(features, labels)
        for i in range(self.max_iters):
            # 选取初始的 alpha1,alpha2
            a1, a2 = self.init_alpha()
            # 边界
            if self.y_train[a1] == self.y_train[a2]:
                L = max(0, a1 + a2 - self.C)
                H = min(self.C, a1 + a2)
            else:
                L = max(0, a2 - a1)
                H = min(self.C, self.C + a2 - a1)

            E1 = self.E[a1]
            E2 = self.E[a2]

            a2_new = self.cut_along_condition(L, H, E1, E2, a1, a2)

            if not a2_new:
                continue
            a1_new = self.alpha[a1] + self.y_train[a1] * self.y_train[a2] * (self.alpha[a2] - a2_new)

            b1_new = self.cal_b_new(E1, a1, a2, a1_new, a2_new)
            b2_new = self.cal_b_new(E2, a2, a1, a2_new, a1_new)

            if 0 < a1_new < self.C:
                b_new = b1_new
            elif 0 < a2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            self.b = b_new
            self.alpha[a1] = a1_new
            self.alpha[a2] = a2_new
            self.E[a1] = self.cal_E(a1)
            self.E[a2] = self.cal_E(a2)

        return 'train down'

    def predict(self, target):
        alpha_y = np.multiply(self.alpha, self.y_train)
        K = self.kernal(self.x_train, target)
        return 1 if np.sum(np.dot(alpha_y, K)) + self.b > 0 else -1

    def score(self, X, Y):
        preds = np.zeros(X.shape[0])
        for i in range(len(X)):
            preds[i] = self.predict(X[i])
        return preds[Y == preds].shape[0] / preds.shape[0]


if __name__ == '__main__':
    # 切分数据
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    idxs = list(range(X.shape[0]))
    random.shuffle(idxs)
    X = X[idxs]
    Y = Y[idxs]
    clf = SVM(max_iters=50, kernal='linear')
    clf.fit(X, Y)
    print(clf.E)
    print(clf.alpha)
    print(clf.score(X, Y))

    # sklearn 实例
    # from sklearn.svm import SVC
    # svm2 = SVC()
    # svm2.fit(X,Y)
    # print(svm2.score(X,Y))










