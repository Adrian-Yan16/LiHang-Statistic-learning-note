import numpy as np


class LeastSqrtree:
    def __init__(self,X_train,y_train,epsilon):
        self.X_train = X_train
        self.y_train = y_train
        self.epsilon = epsilon
        self.feature_count = X_train.shape[1]
        self.tree = None

    def devide(self,X,y,feature_count):
        # 初始化损失列表
        costs = np.zeros((feature_count,len(X)))
        for i in range(feature_count):
            for k in range(len(X)):
                value = X[k,i]
                y1 = y[np.where(X[:,i] <= value)]
                y2 = y[np.where(X[:,i] > value)]
                c1 = np.mean(y1)
                c2 = np.mean(y2)
                y1 = y1 - c1
                y2 = y2 - c2
                costs[i,k] = np.sum(y1 ** 2) + np.sum(y2 ** 2)
        # 选取最优切分特征及切分点
        cost_index = np.where(costs == np.min(costs))
        j = cost_index[0][0]
        s = cost_index[1][0]
        # 两个区域的均值
        c1 = np.mean(y[np.where(X[:,j]<=X[s,j])])
        c2 = np.mean(y[np.where(X[:,j]>X[s,j])])
        return j,s,costs[cost_index],c1,c2

    def train(self,x,y,feature_count):
        j,s,min_val,c1,c2 = self.devide(x,y,feature_count)
        # 初始化树
        tree = {'feature':j,'value':x[s,j],'left':None,'right':None}
        if min_val < self.epsilon or len(y[np.where(x[:,j] <= x[s,j])]) <= 1:
            tree['left'] = c1
        else:
            tree['left'] = self.train(x[np.where(x[:,j] <= x[s,j])],
                                      y[np.where(x[:,j] <= x[s,j])],
                                      self.feature_count)
        if min_val < self.epsilon or len(y[np.where(x[:,j] > x[s,j])]) <= 1:
            tree['right'] = c2
        else:
            tree['right'] = self.train(x[np.where(x[:, j] > x[s, j])],
                                      y[np.where(x[:, j] > x[s, j])],
                                      self.feature_count)
        return tree

    def fit(self):
        self.tree = self.train(self.X_train,self.y_train,self.feature_count)


if __name__ == '__main__':
    train_X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
    y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])
    model = LeastSqrtree(train_X,y,0.2)
    model.fit()
    print(model.tree)



















