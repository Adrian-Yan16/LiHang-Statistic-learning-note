import numpy as np

X = np.arange(1,11,1)
Y = [5.56,5.7,5.91,6.4,6.8,7.05,8.90,8.70,9.00,9.05]

def fit(X,Y,batch = 7):
    length = len(X)
    # 8.3.1 初始化 f0(x) = 0
    Tx = np.zeros_like(Y)
    min_ms = float('INF')
    best_split_point = 0
    c1 = 0
    c2 = 0
    for b in range(batch):
        # 8.3.2.a 计算残差以及Ti(x)
        for i in range(length):
            if X[i] <= best_split_point:
                Y[i] -= c1
                Tx[i] += c1
            else:
                Y[i] -= c2
                Tx[i] += c2
        print('T%d(x):'%(b),Tx)
        for i in range(length - 1):
            split_point = (X[i] + X[i + 1]) / 2
            # 切分点前后数据的均值
            tmp_c1 = round(sum(Y[:i + 1]) / (i + 1),2)
            tmp_c2 = round(sum(Y[i+1:]) / (length - (i + 1)),2)
            # 平方损失误差
            ms = 0
            for j in range(length):
                if j <= i:
                    ms += (Y[j] - tmp_c1) ** 2
                else:
                    ms += (Y[j] - tmp_c2) ** 2
            if ms < min_ms:
                min_ms = ms
                best_split_point = split_point
                c1 = tmp_c1
                c2 = tmp_c2

        print('平方损失误差:',round(min_ms,2))
        min_ms = float('INF')
        print('最优分割点及c1,c2:',best_split_point,c1,c2)




fit(X,Y)



