from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,2],
              [2,3],
              [3,3],
              [2,1],
              [3,2]])
y = np.array([1,1,1,-1,-1])

clf = SVC(kernel='linear')
clf.fit(x,y)
w = clf.coef_
b = clf.intercept_
support_vecs = clf.support_vectors_

print(w)
print(b)
print(support_vecs)

# 绘制数据点
color_seq = ['red' if v == 1 else 'blue' for v in y]
plt.scatter([i[0] for i in x], [i[1] for i in x], c=color_seq)
# 得到x轴的所有点
xaxis = np.linspace(0, 3.5)
w = clf.coef_[0]
# 计算斜率
a = -w[0] / w[1]
# 得到分离超平面
y_sep = a * xaxis - (clf.intercept_[0]) / w[1]
# 下边界超平面
b = clf.support_vectors_[0]
yy_down = a * xaxis + (b[1] - a * b[0])
# 上边界超平面
b = clf.support_vectors_[-1]
yy_up = a * xaxis + (b[1] - a * b[0])
# 绘制超平面
plt.plot(xaxis, y_sep, 'k-')
plt.plot(xaxis, yy_down, 'k--')
plt.plot(xaxis, yy_up, 'k--')
# 绘制支持向量
plt.xlabel('$x^{(1)}$')
plt.ylabel('$x^{(2)}$')
plt.scatter(clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors='none',
            edgecolors='k')
plt.show()
