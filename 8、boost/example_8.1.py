import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt

# 加载训练数据
X = np.array([[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2],
              [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]])

y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])


class AdaBoost:
    def __init__(self, X, y, tol=0.05, max_iter=10):
        self.X = X
        self.y = y
        self.tol = tol
        self.max_iter = max_iter
        self.w = np.ones_like(X[:,1]) / X.shape[0]
        self.G = []

    def build_stump(self):
        m,n = self.X.shape
        e_min = np.inf
        # 分类标签
        sign = None
        best_stump = {}
        for i in range(n):
            sorted_X = sorted(set(X[:,i]))
            for j in range(len(sorted_X)):
                if j == 0:
                    thresh = sorted_X[0] - 0.1
                else:
                    thresh = (sorted_X[j] + sorted_X[j - 1]) / 2
                for inequal in ['lt','rt']:
                    pred = self.base_estimator(self.X,i,thresh,inequal)
                    error = sum((pred != y) * self.w)
                    # print(error)
                    if error < e_min:
                        e_min = error
                        sign = pred
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh
                        best_stump['ineq'] = inequal
        return best_stump,sign,e_min

    def base_estimator(self,X,dim,thresh,inequal):
        pred = np.ones(X.shape[0])
        if inequal == 'lt':
            pred[X[:,dim] > thresh] = -1
        else:
            pred[X[:,dim] < thresh] = -1
        return pred

    def updata_w(self,alpha,pred):
        temp_w = self.w * np.exp(-alpha * self.y * pred)
        self.w = temp_w / sum(temp_w)

    def fit(self):
        G = 0
        for i in range(self.max_iter):
            best_stump,sign,error = self.build_stump()
            alpha = 0.5 * np.log((1-error)/error)
            best_stump['alpha'] = alpha
            self.G.append(best_stump)
            G += alpha * sign
            new_pred = np.sign(G)
            error = sum(new_pred != self.y) / self.y.shape[0]
            print('预测错误样本率为：', error)
            print('预测结果：', new_pred)
            if error < self.tol:
                break
            else:
                self.updata_w(alpha,new_pred)

    def predict(self,X):
        m = X.shape[0]
        G = np.zeros(m)
        for i in range(len(self.G)):
            stump = self.G[i]
            _G = self.base_estimator(X,stump['dim'],stump['thresh'],stump['ineq'])
            alpha = stump['alpha']
            G += alpha * _G
        pred = np.sign(G)
        return pred.astype(int)

    def score(self,X,y):
        pred = self.predict(X)
        error = sum(pred != self.y) / self.y.shape[0]
        return 1 - error

if __name__ == '__main__':

    # clf = AdaBoost(X, y)
    # clf.fit()
    # y_predict = clf.predict(X)
    # score = clf.score(X, y)
    # print("预测正确率：{:.2%}".format(score))

    def create_data():
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['label'] = iris.target
        df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
        data = np.array(df.iloc[:100, [0, 1, -1]])
        for i in range(len(data)):
            if data[i, -1] == 0:
                data[i, -1] = -1
        # print(data)
        return data[:, :2], data[:, -1]


    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    plt.scatter(X[:50, 0], X[:50, 1], label='0')
    plt.scatter(X[50:, 0], X[50:, 1], label='1')
    plt.legend()

    adaboost = AdaBoost(X,y)
    adaboost.fit()





















