import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:, :-1], data[:, -1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class GaussianNB:
    def __init__(self):
        self.model = None

    # 期望
    def mean(self,X):
        return sum(X) / len(X)

    # 标准差
    def stdev(self,X):
        mean = self.mean(X)
        return math.sqrt(sum([pow(x-mean,2) for x in X])/len(X))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理 X
    def summarize(self,X):
        X_train = [(self.mean(x),self.stdev(x)) for x in zip(*X)]
        return X_train

    # 分类别求出数学期望和标准差
    def fit(self,X,y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f,label in zip(X,y):
            data[label].append(f)
        self.model = {
            label:self.summarize(X_train) for label,X_train in data.items()
        }
        return 'train compelete'

    # 求概率
    def cal_probability(self,data):
        probability = {}
        for label,value in self.model.items():
            probability[label] = 1
            for i in range(len(value)):
                mean,stdev = value[i]
                probability[label] *= self.gaussian_probability(data[i],mean,stdev)
        return probability

    def predict(self,x):
        label = sorted(self.cal_probability(x).items(),key=lambda x:x[-1])[-1][0]
        return label

    def score(self,x,y):
        right = 0
        for X,Y in zip(x,y):
            label = self.predict(X)
            if label == Y:
                right += 1
        return right / float(len(x))


if __name__ == '__main__':
    nb = GaussianNB()
    print(nb.fit(X_train,y_train))
    print(nb.predict([4.4, 3.2, 1.3, 0.2]))
    print(nb.score(X_test,y_test))














