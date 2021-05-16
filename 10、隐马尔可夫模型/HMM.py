import numpy as np
import random


class HiddenMarkov:
    def __init__(self):
        self.alphas = None
        self.betas = None
        self.A = None
        self.B = None
        self.Pi = None

    # 前向后向算法求概率计算问题 即P(O|lambda)
    def forward(self,N,V,A,B,Pi,O):
        # N = len(Q)
        # T 为 观测序列 O 的长度
        T = len(O)
        self.alphas = np.zeros((T,N))
        for t in range(T):
            idx_o_t = list(V).index(O[t])
            # 计算 alpha 的初值
            if t == 0:
                for i in range(N):
                    self.alphas[t][i] = Pi[i] * B[i][idx_o_t]
            # 计算其他时刻 alpha 的值
            else:
                for j in range(N):
                    self.alphas[t][j] = np.dot(self.alphas[t-1],A[:,j]) * B[j][idx_o_t]
        # P(O|lambda) 的概率为 alpha 在 T 时刻的和
        return np.sum(self.alphas[-1])

    def backward(self,N,V,A,B,Pi,O):
        # N = len(Q)
        # T 为观测序列 O 的长度
        T = len(O)
        self.betas = np.zeros((T, N))
        for t in range(T-1,-1,-1):
            # 初始化
            if t == T - 1:
                for i in range(N):
                    self.betas[t][i] = 1
            # 递推
            else:
                idx_o_t = list(V).index(O[t + 1])
                for i in range(N):
                    self.betas[t][i] = np.dot(np.multiply(A[i],B[:,idx_o_t]),self.betas[t + 1])
        idx_o_t = list(V).index(O[0])
        return np.dot(np.multiply(Pi,B[:,idx_o_t]),self.betas[0])

    def init_args(self,N,M):
        # 随机生成模型，但要满足约束条件
        for i in range(N):
            random_list = [random.randint(0,100) for i in range(N)]
            sum_random = sum(random_list)
            for j in range(N):
                self.A[i][j] = random_list[j] / sum_random
        for i in range(N):
            random_list = [random.randint(0,100) for i in range(M)]
            sum_random = sum(random_list)
            for j in range(M):
                self.B[i][j] = random_list[j] / sum_random

        random_list = [random.randint(0, 100) for i in range(N)]
        sum_random = sum(random_list)
        for i in range(N):
            self.Pi[i] = random_list[i]/sum_random

    def cal_gamma(self,i,t):
        numerator = self.alphas[t][i] * self.betas[t][i]
        denomicator = np.dot(self.alphas[t],self.betas[t])
        return numerator/denomicator

    def cal_xi(self,i,j,t,idx_o_t):
        numerator = self.alphas[t][i] * self.A[i][j] \
                    * self.B[j][idx_o_t] * self.betas[t+1][j]
        denomicator = np.dot(np.multiply(self.alphas[t],self.alphas[i]),
                             np.multiply(self.B[j],self.betas[t+1]))
        return numerator/denomicator

    # Beum-Welch（非监督学习）求模型参数
    def BeumWelch(self,O,iters,N,M):
        T = len(O)
        self.init_args(N,M)

        for iter in range(iters):
            print('迭代次数：',iter)
            self.forward(N,set(O),self.A,self.B,self.Pi,O)
            self.backward(N,set(O),self.A,self.B,self.Pi,O)
            # A
            for i in range(N):
                for j in range(N):
                    numerator = 0.0
                    denomicator = 0.0
                    for t in range(T-1):
                        numerator += self.cal_gamma(i,t)
                        denomicator += self.cal_xi(i,j,t,O[t+1])
                    self.A[i][j] = numerator / denomicator
            # B
            for j in range(N):
                for k in range(M):
                    numerator = 0.0
                    denomicator = 0.0
                    for t in range(T-1):
                        value = self.cal_gamma(j,t)
                        if k == O[t]:
                            numerator += value
                        denomicator += value
                    self.B[j][k] = numerator / denomicator
            # Pi
            for i in range(N):
                self.Pi[i] = self.cal_gamma(i,0)

    # 随机生成状态序列，观测序列
    def generate(self,T):
        S = []
        # 生成初始状态
        ran = random.randint(0,1000)/1000.0
        i = 0
        while self.Pi[i] < ran or self.Pi[i] < 0.0001:
            ran -= self.Pi[i]
            i += 1
        S.append(i)

        # 生成状态序列
        for i in range(1,T):
            last = S[-1]
            ran = random.randint(0,1000)/1000.0
            i = 0
            while self.A[last][i] < ran or self.A[last][i] < 0.0001:
                ran -= self.A[last][i]
                i += 1
            S.append(i)

        # 生成观测序列
        O = []
        for i in range(T):
            k = 0
            ran = random.randint(0,1000)/1000.0
            while self.B[S[i]][k] < ran or self.B[S[i]][k] < 0.0001:
                ran -= self.B[S[i]][k]
                k += 1
            O.append(k)
        return S,O

    # 预测问题
    def viterbi(self, Q, V, A, B,Pi,O):
        N = len(Q)
        T = len(O)
        deltas = np.zeros((T, N))
        Psis = np.zeros((T,N))
        for t in range(T):
            for i in range(N):
                idx_o_t = list(V).index(O[t])
                # 初始化
                if t == 0:
                    deltas[t][i] = Pi[i] * B[i][idx_o_t]
                    Psis[t][i] = 0
                # 递推
                else:
                    temp = [(deltas[t - 1][j] * A[j][i], j) for j in range(N)]
                    deltas[t][i] = max(temp, key=lambda x:x[0])[0] * B[i][idx_o_t]
                    Psis[t][i] = max(temp,key=lambda x:x[0])[1]
        P = max(deltas[-1])
        path = []
        s_T = np.argmax(deltas[-1])
        path.insert(0,s_T)
        for t in range(T - 2,-1,-1):
            s = Psis[t + 1][path[0]]
            path.insert(0,int(s))
        return P,[i + 1 for i in path]


if __name__ == '__main__':
    Q = np.array([1, 2, 3])
    V = np.array(['红', '白'])
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    O = np.array(['红', '白', '红', '白',])
    PI = np.array([0.2, 0.4, 0.4])
    hmm = HiddenMarkov()
    # print(hmm.forward(Q,V,A,B,PI,O))
    # print(hmm.backward(Q,V,A,B,PI,O))
    # print(hmm.alphas)
    # print(hmm.betas)
    print(hmm.viterbi(Q,V,A,B,PI,O))



