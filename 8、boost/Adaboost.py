import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier


class BasicModel:
    def __init__(self):
        self.best_split_point = None
        # positive为True，代表小于阈值，预测为 1
        self.positive = None

    def fit(self,X,Y,W):
        error = np.sum(W)
        sort_X = list(X) + [max(X) + 1] + [min(X) - 1]
        sort_X.sort()
        # 计算最优切分点
        for i in range(len(sort_X) - 1):
            split_point = (sort_X[i] + sort_X[i + 1]) / 2
            temp_pred = np.ones_like(Y)
            temp_pred[X > split_point] = 1
            temp_pred[X < split_point] = -1
            temp_error = np.sum((temp_pred != Y) * W)
            if temp_error < error:
                self.best_split_point = split_point
                self.positive = True
                error = temp_error
            temp_pred[X < split_point] = 1
            temp_pred[X > split_point] = -1
            temp_error = np.sum((temp_pred != Y) * W)
            if temp_error < error:
                self.best_split_point = split_point
                self.positive = False
                error = temp_error

    def predict(self,X):
        pred = np.ones_like(X)
        if self.positive:
            pred[X > self.best_split_point] = -1
        else:
            pred[X < self.best_split_point] = -1
        return pred

class Adaboost:
    def __init__(self,batch = 10):
        self.batch = batch
        self.alphas = []
        self.best_models = []

    def fit(self,X,Y):
        N = len(X)
        w = np.ones((N,)) / N
        for i in range(self.batch):
            model = BasicModel()
            model.fit(X,Y,w)
            pred = model.predict(X)
            self.best_models.append(model)
            # 计算分类误差率
            e_m = np.sum((pred != Y) * w)
            # 计算 alpha
            alpha_m = 0.5 * np.log((1-e_m)/e_m)
            self.alphas.append(alpha_m)
            temp_w = w * np.exp(-alpha_m * Y * pred)
            # 计算 w_{m+1}
            w = temp_w / np.sum(temp_w)
            new_pred = self.predict(X)
            error = sum(new_pred != Y)
            print('预测错误样本数为：',error)
            print('预测结果：',new_pred)
            if error == 0:
                break
        print("*" * 50)
        print("模型训练结束, 基学习器个数%d" % len(self.alphas))
        print("各个基学习器权重:", self.alphas)
        print("各个基学习器分割点及其正例反向（left表示小于分割点为正）")
        for model in self.best_models:
            print(model.best_split_point, "left" if model.positive else "right")
        print("*" * 50)

    def predict(self,X):
        pred = np.zeros_like(X) * 1.0
        for alpha,best_model in zip(self.alphas,self.best_models):
            temp_pred = best_model.predict(X)
            pred += alpha * temp_pred
        pred[pred > 0] = 1
        pred[pred < 0] = -1
        return pred

    
if __name__ == '__main__':
    # X = np.arange(10)
    # Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    # adaboost = Adaboost(4)
    # adaboost.fit(X,Y)
    # data
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

    adaboost = Adaboost(10)
    adaboost.fit(X,y)

    # sklearn
    ada = AdaBoostClassifier()
    adaboost.fit(X,y)























