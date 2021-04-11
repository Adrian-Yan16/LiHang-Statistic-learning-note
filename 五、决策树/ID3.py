import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from math import log
import pprint


class Node:
    def __init__(self,root=True,label=None,feature=None):
        self.root = root
        self.label = label
        self.feature = feature
        self.tree = {}

    def add_node(self,val,node):
        self.tree[val] = node

    def predict(self,features):
        if self.root:
            return self.label
        try:
            return self.tree[features[self.feature]].predict(features)
        except:
            return

class ID3:
    def __init__(self,epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    def cal_H_D(self,LabelArr):
        # 经验熵
        H_D = 0
        labels = set(LabelArr)
        for label in labels:
            p = LabelArr[LabelArr==label].size / LabelArr.size
            H_D += -p * np.log2(p)
        return H_D

    def cal_H_D_A(self,A_features,LabelArr):
        """
        经验条件熵
        :param A_features: A=a 时的特征
        :param LabelArr: 标签
        :return:
        """
        H_D_A=0
        features = set(A_features)
        for feature in features:
            H_D_A += -A_features[A_features==feature].size / A_features.size * self.cal_H_D(LabelArr[A_features==feature])
        return H_D_A

    def best_feature(self,dataset,LabelArr):
        """
        找到信息增益最大的特征
        :param dataset: 特征值
        :param LabelArr: 标签
        :return:
        """
        H_D = self.cal_H_D(LabelArr)
        min_H_D_A = float('inf')
        Ag = -1
        for i in range(dataset.iloc[0].size):
            A_features = dataset.iloc[:,i]
            H_D_A = -self.cal_H_D_A(A_features,LabelArr)
            if H_D_A < min_H_D_A:
                min_H_D_A = H_D_A
                Ag = i
        return Ag, H_D - min_H_D_A

    def train(self,train_data):
        """
        训练，构建决策树过程
        :param train_data: DataFrame类型
        :return:
        """
        y_train,features = train_data.iloc[:,-1],train_data.iloc[:,:-1]
        value_count = y_train.value_counts()
        # 1，如果 D 中实例均为同一类，则 T 为单结点树，将类 Ck 作为该结点的类标记
        if value_count.size == 1:
            return Node(root=True,label=y_train.iloc[0])
        # 2，若 A 为空，则 T 为单结点树，将 D 中实例数最大的类 Ck 作为该结点的类标记
        if features.size == 0:
            return Node(root=True,label=
                        value_count.sort_values(ascending=False).index[0])
        # 3，计算信息增益，得到最优特征 Ag
        Ag,info_gain = self.best_feature(features,y_train)
        max_feature = features.iloc[:,Ag]
        # 4，Ag 的信息增益小于 epsilon，则将 T 置为单结点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if info_gain < self.epsilon:
            return Node(root=True,label=
                        value_count.sort_values(ascending=False).index[0])
        # 5，构建 Ag 子集
        node_tree = Node(root=False,feature=Ag)
        feature_list = set(max_feature)
        for feature in feature_list:
            try:
                sub_train = train_data[max_feature==feature].drop(Ag,axis=1)
            except:
                sub_train = train_data[max_feature == feature]
            sub_tree = self.train(sub_train)
            node_tree.add_node(feature,sub_tree)
        return node_tree

    def fit(self,train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self,X_test):
        return self._tree.predict(X_test)

    def score(self,X_test,y_test):
        preds = []
        for x in X_test:
            pred = self.predict(x)
            preds.append(pred)
        right = 0
        for i in range(len(preds)):
            if preds[i] == y_test[i]:
                right += 1
        return right / len(preds)


def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

# data
def create_iris():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:, :2], data[:, -1]


X, y = create_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


if __name__ == '__main__':
    datasets,labels = create_data()
    data_df = pd.DataFrame(datasets)
    dt = ID3()
    # tree = dt.fit(data_df)
    # print(dt.predict(['老年', '否', '否', '一般']))

    train_data = np.column_stack([X_train,y_train])
    tree = dt.fit(pd.DataFrame(train_data))
    print('训练完成')
    print('myself:',dt.score(X_test,y_test))

















