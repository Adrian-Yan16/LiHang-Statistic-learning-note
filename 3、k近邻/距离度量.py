import math

# 距离度量
def cal_L(x1,x2,p=2):
    # p = 1 为曼哈顿距离
    # p = 2 为欧式距离
    # p = 3 为切比雪夫距离
    if len(x1) == len(x2) and len(x1) > 0:
        sum = 0
        for i in range(len(x1)):
            sum += math.pow(abs(x1[i] - x2[i]),p)
        return math.pow(sum,1/p)
    return 0


if __name__ == '__main__':
    x1 = [1,1]
    x2 = [5,1]
    x3 = [4,4]

    for i in range(1,5):
        r = {'1-{}'.format(c):cal_L(x1,c,p=i) for c in [x2,x3]}
        print(min(zip(r.values(),r.keys())))