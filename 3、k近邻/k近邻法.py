import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# 数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class KNN:
    def __init__(self,x_train,y_train,k_neighbors=3,p=2):
        self.X_train = x_train
        self.Y_train = y_train
        self.k = k_neighbors
        self.p = p

    def cal_L(self,x1, x2, p=2):
        # p = 1 为曼哈顿距离
        # p = 2 为欧式距离
        # p = 3 为切比雪夫距离
        if len(x1) == len(x2) and len(x1) > 0:
            sum = 0
            for i in range(len(x1)):
                sum += math.pow(abs(x1[i] - x2[i]), p)
            return math.pow(sum, 1 / p)
        return 0

    def predict(self,x):
        k_list = []
        # 选 k 个点
        for i in range(self.k):
            # 计算距离并保存
            # dist = self.cal_L(x,self.X_train[i])
            dist = np.linalg.norm(x - self.X_train[i],ord=self.p)
            k_list.append((dist,self.Y_train[i]))

        for i in range(self.k,len(self.X_train)):
            # 找到离 x 最近的 k 个点
            max_index = k_list.index(max(k_list,key=lambda x:x[0]))
            # dist = self.cal_L(x,self.X_train[i])
            dist = np.linalg.norm(x - self.X_train[i],ord=self.p)
            if k_list[max_index][0] > dist:
                k_list[max_index] = (dist,self.Y_train[i])

        # 统计这 k 个点的类别，最多的即为 x 的类别
        knn = [k[-1] for k in k_list]
        count_pairs = Counter(knn)
        print(count_pairs.items())
        pred_class = sorted(count_pairs.items(),key=lambda x:x[1])[-1][0]
        return pred_class

    # 打分
    def score(self,x_test,y_test):
        right_pred = 0
        for x,y in zip(x_test,y_test):
            pred = self.predict(x)
            if pred == y:
                right_pred += 1
        return right_pred / len(y_test)


clf = KNN(X_train,y_train,p=1)
print(clf.score(X_test,y_test))

test_point = np.array([[6.0, 3.0]])
print('Test Point: {}'.format(clf.predict(test_point)))

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# sk-learn 实例
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
# n_neighbors: 临近点个数
# p: 距离度量
# algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}
# weights: 确定近邻的权重
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

























