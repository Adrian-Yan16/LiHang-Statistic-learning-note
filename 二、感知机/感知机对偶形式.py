import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]
df.label.value_counts()



data = np.array(df.iloc[:100, [0, 1, -1]])
x, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])


class Perceptron:
    def __init__(self,X_train,Y_train,l_rate=0.1):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N = len(self.X_train)
        self.a = np.zeros(self.N)
        self.b = 0
        self.l_rate = l_rate
        self.cal_gram()

    def cal_gram(self):
        self.gram = np.empty((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                # G = [xi * xj]
                self.gram[i][j] = np.dot(self.X_train[i],self.X_train[j])

    def judge(self,i):
        res = np.dot(self.a * y,self.gram[:,i])
        res = (res + self.b)*y[i]
        return res

    def fit(self):
        false = True
        while false:
            false_count = 0
            for d in range(self.N):
                if self.judge(d) <= 0:
                    self.update(d)
                    false_count += 1
            if false_count == 0:
                false = False
        print('Perceptron Model fit complete')

    def update(self,i):
        self.a[i] = self.a[i] + self.l_rate
        self.b = self.b + self.l_rate * self.Y_train[i]

    def cal_w(self):
        w = np.dot(self.a * self.Y_train, self.X_train)
        return w



percep = Perceptron(x,y)
percep.fit()

x = np.linspace(4,7,10)
w = percep.cal_w()
y_ = -(w[0] * x + percep.b)/w[1]
plt.plot(x,y_)
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
























