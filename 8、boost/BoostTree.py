import numpy as np

X = np.arange(1,11,1)
Y = np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.90,8.70,9.00,9.05])


class RegressionBoostTree:
    def __init__(self,M):
        self.M = M

    def fit(self,X,Y):
        length = len(X)
        # 8.3.1 初始化 f0(x) = 0
        self.Tx = np.zeros_like(Y)
        min_ms = float('INF')
        self.best_split_point = [0] * (self.M + 1)
        c1 = 0
        c2 = 0
        for m in range(1,self.M + 1):
            # 8.3.2.a 计算残差以及Ti(x)
            Y[X < self.best_split_point[m - 1]] -= c1
            self.Tx[X < self.best_split_point[m - 1]] += c1
            Y[X > self.best_split_point[m - 1]] -= c2
            self.Tx[X > self.best_split_point[m - 1]] += c2
            print('T%d(x):'%(m),self.Tx)
            for i in range(length - 1):
                split_point = (X[i] + X[i + 1]) / 2
                y1 = Y[X < split_point]
                y2 = Y[X > split_point]
                # 切分点前后数据的均值
                tmp_c1 = round(float(np.mean(y1)),2)
                tmp_c2 = round(float(np.mean(y2)),2)
                # 平方损失误差
                ms = np.sum(np.power(y1 - tmp_c1,2)) + \
                     np.sum(np.power(y2 - tmp_c2,2))
                if ms < min_ms:
                    min_ms = ms
                    self.best_split_point[m] = split_point
                    c1 = tmp_c1
                    c2 = tmp_c2
            print('平方损失误差:',round(min_ms,2))
            min_ms = float('INF')
            print('最优分割点及c1,c2:',self.best_split_point[m],c1,c2)

    def predict(self,x):
        best_splits = set(self.best_split_point)
        for split in best_splits:
            if x < split:
                return self.Tx[int(split)]
        return self.Tx[-1]


if __name__ == '__main__':
    rt = RegressionBoostTree(7)
    rt.fit(X,Y)




