import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]
df.label.value_counts()



data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])


class Perceptron:
    def __init__(self,l_rate = 0.1):
        self.w = np.ones(len(data[0]) - 1,dtype=np.float32)
        self.b = 0
        self.l_rate = l_rate

    def judge(self,x,y):
        l = y * (np.dot(self.w,x) + self.b)
        return l

    def fit(self,X_train,Y_train):
        false = True
        max_f = 0
        while false:
            false_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = Y_train[d]
                if self.judge(X,y) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y,X)
                    self.b = self.b + self.l_rate * y
                    false_count += 1
            # 记录误分类点个数
            max_f = max(max_f,false_count)
            if false_count == 0:
                false = False
        print('误分类点有：%d个'%max_f)
        print("Perceptron Model")

    def predict(self,x):
        # 预测
        y = np.dot(self.w,x) + self.b
        return 1 if y >= 0 else -1


percep = Perceptron()
percep.fit(X,y)

x = np.linspace(4,7,10)
y = - (percep.w[0] * x + percep.b) / percep.w[1]

plt.plot(x,y)



# sklearn实例
# from sklearn.linear_model import Perceptron
#
# clf = Perceptron(fit_intercept=True,max_iter=100,shuffle=True)
# clf.fit(X,y)
#
# # 画布大小
# plt.figure(figsize=(10,10))
#
# # 中文标题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.title('鸢尾花线性数据示例')
#
#
# x = np.arange(4,8)
# # coef_——权值，intercept_——截距
# y = -(clf.coef_[0][0] * x + clf.intercept_)/clf.coef_[0][1]
# plt.plot(x,y)

# 拿出iris数据集中两个分类的数据和[sepal length，sepal width]作为特征
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
























