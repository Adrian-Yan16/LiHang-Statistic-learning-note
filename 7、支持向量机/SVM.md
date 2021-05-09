SVM（支持向量机）是一种二分类模型，基本模型为特征空间上间隔最大的线性分类器，间隔最大使它有别于感知机，支持向量机还包括核技巧，这使它成为实质上的非线性分类器；学习策略为间隔最大化，可形式化为凸二次规划（convex quadratic programming），也等价于正则化的合页损失函数最小化问题，学习算法是凸二次规划的最优化算法。

支持向量机学习方法包括构建由简到繁的模型：线性可分支持向量机，线性支持向量机，非线性支持向量机。

当输入空间为欧式空间或离散集合、特征空间为希尔伯特空间时，核函数表示将输入从输入空间映射到输出空间得到的特征向量之间的内积。其等价于隐式的在高维的特征空间中学习线性支持向量机

# 线性可分支持向量机与硬间隔最大化

## 线性可分支持向量机

线性可分支持向量机，线性支持向量机认为输入空间和特征空间的元素一一对应，并将输入空间的输入映射为特征空间的特征向量，非线性支持向量机利用非线性映射将输入映射为特征空间的特征向量，支持向量机的学习是在特征空间中进行的

给定特征空间的数据集 T，$(x_i,y_i)$ 为样本点，假设数据集线性可分

学习目标为找到一个分类超平面 $wx + b = 0$ ，能将实例分到不同的类，分离超平面由法向量 w 和截距 b 决定。法向量指向的一侧为正类，另一侧为负类

**定义：线性可分支持向量机** 给定线性可分数据集，通过间隔最大化或等价地求解相应的凸二次规划问题学习得到的分离超平面为
$$
w^*\cdot x + b^* = 0\tag{7.1}
$$
及相应的决策函数
$$
f(x) = sign(w^*x + b^*)\tag{7.2}
$$
称为线性可分支持向量机

## 函数间隔与几何间隔

一般来说，一个点到超平面的距离可以表示分离预测的确信程度，在分离超平面确定的情况下，$|w\cdot x + b|$ 能相对地表示点到平面的距离，而 $w\cdot x + b$ 的符号与类标记 y 是否一致能够表示预测是否正确，所以可用 $y(w\cdot x + b)$ 来表示分类的正确性及置信度，这就是函数间隔的

**定义：函数间隔** 对于数据集 T 和超平面 （w,b），超平面与样本点 $(x_i,y_i)$ 的函数间隔为
$$
\hat \gamma_i = y_i(w\cdot x_i + b)\tag{7.3}
$$

超平面对数据集 T 的函数间隔为所有样本点的函数间隔的最小值
$$
\hat \gamma = min\hat\gamma_i
$$
但是在选择分离超平面时，只有函数间隔是不够的，因为当 w 和 b 成倍增长时，函数间隔也会成倍增长。因此需要将法向量进行约束，比如规范化，||w|| = 1,使得间隔是确定的，这是函数间隔就是几何间隔

**定义 几何间隔** 超平面对于样本点 $(x_i,y_i)$ 的几何间隔
$$
\gamma_i = y_i(\frac{w}{||w||}\cdot x + \frac b{||w||})\tag{7.4}
$$
超平面对于数据集的几何间隔为所有样本点的几何间隔最小值
$$
\gamma = min\gamma_i
$$
函数间隔与几何间隔的关系为
$$
\gamma_i = \frac{\hat\gamma_i}{||w||}\\
\gamma = \frac{\hat\gamma}{||w||}
$$

||w|| = 1 时，两者相等，如果超平面参数成比例的增长（超平面不变），函数间隔也成比例的改变，几何间隔不变

## 间隔最大化

支持向量机的学习策略是求解能够正确划分训练数据集并且几何间隔最大的分离超平面，对线性可分数据集，这样的超平面有无数个，但是间隔最大的只有一个

间隔最大化的直观解释是：对训练数据集找到几何间隔最大的分离超平面意味着以充分大的确信度对训练数据进行分类，即对最难分的实例点，也有足够大的确信度将它们正确分类，这样的超平面对未知数据也有很好的预测能力

1. 最大间隔分离超平面

   求解最大间隔分离超平面表示为下面的约束最优化问题
   $$
   max_{w,b} \quad\gamma\tag{7.9}\\
   $$

   $$
   s.t.\quad y_i(\frac{w\cdot x_i + b}{||w||}) >= \gamma\tag{7.10}
   $$

   即我们希望最大化几何间隔 $\gamma$，改写为函数间隔
   $$
   max_{w,b}\quad \frac{\hat\gamma}{||w||}\tag{7.11}
   $$

   $$
   s.t.\quad y_i(w\cdot x_i + b) >= \hat\gamma\tag{7.12}
   $$

   函数间隔的取值并不影响最优化问题的解，比如 w 和 b 同时扩大 $\lambda$ 倍变为 $\lambda w,\lambda b$，函数间隔变为 $\lambda\hat\gamma$，这一改变并不影响不等式约束和目标函数的优化，这样就可以取 $\hat\gamma = 1$ ，带入上面的最优化问题，而最大化 $\frac1{||w||}$ 与最小化 $\frac12||w||^2$ 是一样的，所以有
   $$
   min_{w,b}\quad\frac12||w||^2\tag{7.13}\\
   $$

   $$
   s.t. \quad y_i(w\cdot x_i + b) - 1 >= 0\tag{7.14}
   $$

   这是一个凸二次规划问题

   **算法：线性可分支持向量机学习算法——最大间隔法**

   （1）构造并求解约束最优化问题——公式 7.13，7.14，求解做优解 $w^*,b^*$

   （2）由此得到分离超平面 $w^*\cdot x+ b^*=0$ 及分类决策函数 $f(x) = sign(w^*\cdot x+ b^*)$

2. 支持向量和间隔边界

   距离分类超平面最近的样本点为支持向量（support vector），支持向量是使 7.14 成立的点，有
   $$
   H_1:w\cdot x + b = 1\quad y=1\\
   H_2:w\cdot x + b = -1\quad y=-1
   $$
   如图所示，$H_1,H_2$ 上的点即为支持向量

   <img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210423171556123.png" alt="image-20210423171556123" style="zoom:50%;" />

   分离超平面与 $H_1,H_2$ 平行，且位于它们中间，$H_1,H_2$ 之间的距离称为间隔（margin），间隔依赖于法向量 w，等于 $\frac2{||w||}$，$H_1,H_2$ 称为间隔边界

   在决定分离超平面时只有支持向量起作用，支持向量的个数一般很少，所以支持向量机有很少且重要的样本确定

## 学习的对偶算法

线性可分支持向量机的对偶算法（dual algorithm）就是为了求解线性可分支持向量机的最优化问题（7.13~7.14），将它作为原始最优化问题，应用拉格朗日对偶性，通过求解对偶问题得到原始问题（primal problem）的最优解，这样做的优点一是对偶问题往往容易求解，二是自然引入核函数，进而推广到非线性可分数据集

构建拉格朗日函数，对每一个不等式约束 7.14 引入拉格朗日乘子 $\alpha_i >=0$
$$
L(w,b,\alpha) = \frac12||w||^2 - \sum_i\alpha_iy_i(w\cdot x_i + b) + \sum_i\alpha_i\tag{7.18}
$$
$\alpha = (\alpha_1,\alpha_2,\cdots,\alpha_N)^T$ 为拉格朗日乘子向量

根据拉格朗日对偶性，原始问题的对偶问题是极大极小问题：
$$
max_\alpha min_{w,b}L(w,b,\alpha)
$$
所以为了得到对偶问题的解，要先求 w,b 的极小，再求 $\alpha$ 的极大

1. 求 $min_{w,b}L(w,b,\alpha)$

   分别对 w,b 求偏导，得
   $$
   \bigtriangledown_wL = w - \sum_i\alpha_iy_ix_i = 0\\
   \bigtriangledown_bL = \sum_i\alpha_iy_i = 0\\
   \rightarrow w=\sum_i\alpha_iy_ix_i\\
   \sum_i\alpha_iy_i = 0
   $$
   带入兰格朗日函数 7.18，得
   $$
   min_{w,b}L(w,b,\alpha) = -\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_j(x_i\cdot x_j) + \sum_i\alpha_i
   $$

2. 求 $min_{w,b}L(w,b,\alpha)$ 对 $\alpha$ 的极大，即是对偶问题
   $$
   max_\alpha-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_j(x_i\cdot x_j) + \sum_i\alpha_i\tag{7.21}\\
   s.t.\quad\sum_i\alpha_iy_i = 0\\
   \alpha_i \geq 0
   $$
   将 7.21 转化为求极小，就成了下面与之等价的对偶最优化问题：
   $$
   min_\alpha\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_j(x_i\cdot x_j) - \sum_i\alpha_i\tag{7.22}\\
   s.t.\quad\sum_i\alpha_iy_i = 0\\
   \alpha_i \geq 0
   $$

**定理 7.2** 设 $\alpha^* = (\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$ 是对偶问题的最优解，则存在下标 j，使得 $\alpha_j^*>0$，并可按下式求得原始最优化问题的解 $w^*,b^*$
$$
w^* = \sum_i\alpha_i^*y_ix_i\tag{7.25}\\
b^* = y_j - \sum_i\alpha_i^*y_ix_i\cdot x_j
$$
证明：此处 KKT 条件成立，即得
$$
\bigtriangledown_wL = w^* - \sum_i\alpha_i^*y_ix_i = 0\tag{7.27}\\
\bigtriangledown_bL=-\sum_i\alpha_i^*y_i = 0\\
\alpha_i^*(y_i(w^*\cdot x_i + b^*) - 1) = 0\\
y_i(w^*\cdot x_i + b^*) - 1 >= 0\\
\alpha_i^*>=0\\
\rightarrow w^* = \sum_i\alpha_i^*y_ix_i
$$
其中至少有一个 $\alpha_j^*>0$ （假设$\alpha^* = 0$ 由式7.27 可得 $w^* = 0$，w=0 并不是原始问题的最优解<w=0 为什么不是，因为决策函数变为了 f(x) = b，没办法进行分类>，矛盾），对此 j 有
$$
y_j(w^*\cdot x_j + b^*) - 1 = 0\tag{7.28}
$$
将 7.25 代入 7.28 得
$$
b^* = y_j - \sum_i\alpha_i^*y_ix_i\cdot x_j
$$
分离超平面可以写成
$$
\sum_i\alpha_i^*y_ix_i\cdot x_j + b^* = 0\\
f(x) = sign(\sum_i\alpha_i^*y_ix_i\cdot x_j + b^*)
$$
这就是说分类决策函数只依赖于输入 x 和训练样本输入的内积，上式称为线性可分支持向量机的对偶形式

**算法 线性可分支持向量机学习算法**

1. 构造并求解约束最优化问题
   $$
   min_\alpha \sum_i\sum_j\alpha_i\alpha_jy_iy_jx_i\cdot x_j - \sum_i\alpha_i\\
   s.t.\quad \sum_i\alpha_iy_i = 0\\
   \alpha_i>=0
   $$
   求解最优解 $\alpha^* = (\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$

2. 计算
   $$
   w^* = \sum_i\alpha_i^*y_ix_i
   $$
   选择一个正分量 $\alpha_j^* > 0$，求解
   $$
   b^* = y_j - \sum_i\alpha_i^*y_ix_i\cdot x_j
   $$

3. 求得分离超平面及决策函数
   $$
   w^*\cdot x + b^* = 0\\
   f(x) = sign(w^*\cdot x + b^*)
   $$

**定义** **支持向量** 将训练数据集中对应于 $\alpha_i^* > 0$ 的样本 $(x_i,y_i)$ 的实例称为支持向量

根据这一定义，支持向量一定在边界上，有 KKT 互补条件
$$
\alpha_i^*(y_i(w^*\cdot x_i + b) - 1) = 0
$$
对 $\alpha_i^* > 0$ 的实例，有
$$
y_i(w^*\cdot x_i + b^*) = 1\\
或\quad w^*\cdot x_i + b^* = \pm 1
$$

 例：正例点：$x_1(3,3),x_2(4,3)$，负例点：$x_3(1,1)$，利用上述算法求解线性可分支持向量机

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210424094629487.png" alt="image-20210424094629487" style="zoom:50%;" />

（1）求解 $max_\alpha(x_i\cdot x_j 为内积，np.dot(x_i,x_j))$
$$
max_\alpha \frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_jx_i\cdot x_j - \sum_i\alpha_i\qquad\qquad\qquad\qquad\qquad\qquad\quad\\
=18\alpha_1^2 + 25\alpha_2^2 + 2\alpha_3^3 +42\alpha_1\alpha_2 - 12\alpha_1\alpha_3 - 14\alpha_2\alpha_3 - \alpha_1-\alpha_2 - \alpha_3\\
s.t.\quad \sum_i\alpha_iy_i = 0,\alpha_i > =0\\
\rightarrow \alpha_1 + \alpha_2 - \alpha_3 = 0\\
\rightarrow \alpha_3 = \alpha_1 + \alpha_2\quad 代入目标函数\\
\rightarrow s(\alpha_1,\alpha_2) = 4\alpha_1^2 + \frac{13}2\alpha_2^2 + 10\alpha_1\alpha_2 - 2\alpha_1-2\alpha_2
$$
​	分别对 $\alpha_1,\alpha_2$ 求导，可得 $s(\alpha_1,\alpha_2)$ 在 $(\frac{13}2,-1)^T$ 处取极值，但是不满足 $\alpha_i >= 0$ 的条件，所以最小值应在边界上达到

​	当 $\alpha_1=0$，求解 $\bigtriangledown_{\alpha_2} s = 13\alpha_2 - 2 = 0\rightarrow s(0,\frac2{13}) = -\frac2{13}$

​	当 $\alpha_2=0$，$\bigtriangledown_{\alpha_1} s = 8\alpha_1  - 2\alpha_1 = 0 \rightarrow s(\frac14,0)=-\frac14$

​	所以 s 在 $(\frac14,0)$ 处取得最小值，$\alpha_3 = \alpha_1 + \alpha_2 = \frac14$

​	根据公式 7.25 可得
$$
w^* = \sum_i\alpha_i^*y_ix_i = \frac14*1*(3,3)^T + \frac14 * -1*(1,1)^T = (\frac12,\frac12)\\
找一个 \alpha_i^* > 0 的样本点[(1,1),-1]带入得\\
b^* = y_j - \sum_i\alpha_i^*y_ix_i\cdot x_j\qquad\qquad\qquad\qquad\qquad\qquad\qquad\\
=-1 - \frac14*1*(3,3)\cdot(1,1) - \frac14*-1*(1,1)\cdot(1,1)\\
=-2\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad
$$
​	分离超平面及决策函数分别为

​	
$$
\frac12x^1 + \frac12x^2 -2 = 0\\
f(x) = sign(\frac12x^1 + \frac12x^2 -2)
$$

# 线性支持向量机和软间隔最大化

## 线性支持向量机

给定一个线性不可分的训练数据集，也就是，训练数据集中有一些特异点（outlier），将这些特异点取出后，剩下的样本的组成的集合是线性可分的

线性不可分意味着某些样本点不能满足函数间隔大于等于 1 的约束条件 7.14，为了解决这个问题，可以给每个样本的引入一个松弛变量 $\xi\geq0$，使函数间隔加上松弛变量大于等于 1，约束条件变为了
$$
y_i(wx_i + b) \geq 1 - \xi_i
$$
对每个松弛变量，支付一个代价，目标函数变为了
$$
\frac12||w||^2 + C\sum_i\xi_i\tag{7.31}
$$
C > 0 表示惩罚参数，由应用问题决定，C 大时对误分类的惩罚增大，C 小时则减小，最小化目标函数 7.31 有两层含义，一是使几何间隔最小，而是使误分类个数尽量小，C 是调和二者的系数

线性支持向量机的凸二次规划问题为
$$
min_{w,b} \frac12||w||^2 + C\sum_i\xi_i\\
s.t.\quad y_i(wx_i + b) \geq 1-\xi_i\\
\xi_i \geq 0
$$

## 学习的对偶算法

原始最优化问题的拉格朗日函数为
$$
L(w,b,\xi,\alpha,\mu)=\frac12||w||^2 + C\sum_i\xi_i -\sum_i\alpha_i[y_i(wx_i + b) - 1 + \xi_i] -\sum_i\mu_i\xi_i\\
\rightarrow\bigtriangledown_wL = w - \sum_i\alpha_iy_ix_i = 0\\
\bigtriangledown_bL=-\sum_i\alpha_iy_i = 0\\
\bigtriangledown_{\xi_i}L = C - \alpha_i-\mu_i = 0\\
$$
带入原始函数得
$$
min_{w,b}L(w,b,\xi,\alpha,\mu) = -\frac12\sum_i\alpha_i\alpha_jy_iy_jx_i\cdot x_j + \sum_i\alpha_i\\
s.t.\quad\sum_i\alpha_iy_i = 0\\
\alpha_i\geq0,\mu_i\geq0\\
C - \alpha_i - \mu_i = 0
$$
利用等式消去 $\mu_i$ 可得 $0\leq\alpha_i\leq C$，转为极小化后，对偶问题为
$$
min_\alpha\ \frac12\sum_i\alpha_i\alpha_jy_iy_jx_i\cdot x_j - \sum_i\alpha_i\\
s.t.\quad\sum_i\alpha_iy_i = 0\\
0\leq\alpha_i\leq C
$$
**算法 线性支持向量机**

（1）选择惩罚参数 C > 0，求解 $min_\alpha$
$$
min_\alpha\quad\frac12\sum_i\alpha_i\alpha_jy_iy_jx_i\cdot x_j - \sum_i\alpha_i\\
s.t.\quad\sum_i\alpha_iy_i=0\\
0\leq\alpha_i\leq C
$$
​	求解 $\alpha^*=(\alpha_1^*,\alpha_2^*\cdots)^T$

（2）求解 $min_{w,b}$
$$
w^* = \sum_i\alpha_i^*y_ix_i\\
$$
​	选择满足条件 $0\leq\alpha_i\leq C$ 的 $\alpha_i^*$，计算
$$
b^* = y_j - \sum_i\alpha_i^*y_ix_i\cdot x_j
$$
（3）求得分离超平面及决策函数
$$
w^*x+b^*=0\\
f(x) = sign(w^*x+b^*)
$$
步骤（2）中，任何一个满足条件的 $\alpha_i^*$ 都可以求出一个 $b^*$，因此 b 的解不唯一，所以一般用所有符合条件的样本点的均值

## 支持向量

线性不可分的情况下，对偶问题的解 $\alpha^*$ 中，$\alpha_i^* > 0$ 的样本点为软间隔支持向量，要比线性可分支持向量机复杂一些，实例到间隔边界的距离为 $\frac{\xi_i}{||w||}$

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210424114716068.png" alt="image-20210424114716068" style="zoom:50%;" />

若 $\alpha_i^* < C，\xi=0$，支持向量恰好落在间隔边界，若 $\alpha_i^* =C,0<\xi_i<1$，则分类正确，$x_i$ 在分离超平面和间隔边界之间，若 $\alpha_i^*=C,\xi_i^*=1$，$x_i$ 在分离超平面上，若 $\alpha_i^*=C,\xi_i >1$，则 $x_i$ 在分离超平面误分的一侧

## 合页损失函数

线性支持向量机还可以解释为，最小化以下目标函数
$$
\sum_i[1-y_i(wx_i + b)]_+ + \lambda||w||^2
$$
第一项表示经验风险，称为合页损失函数（hinge），下标 “+” 表示以下取正值的函数
$$
[z]_+=\begin{cases}z\quad z>0\\
0\quad z \leq 0\end{cases}
$$
当样本点分类正确且函数间隔 $y_i(wx_i  +b)$ 大于 1 时，损失才为 0，否则为 $1-y_i(wx_i + b)$，比如上图中的 x4，虽被正确分类，但损失不为 0

第二项为 L2 范数

**推导** 令 $1-y_i(wx_i  +b) =\xi_i，\xi_i\geq0$ ，则 $y_i(wx_i+b)\geq1$，于是 $w,b,\xi_i$ 满足原始最优化问题的约束条件，有 $[1-y_i(wx_i + b)]_+ = [\xi_i]_+=\xi_i$，最优化问题可写成
$$
min_{w,b} \sum_i\xi_i + \lambda||w||^2
$$
取 $\lambda=\frac1{2C}$ ，有
$$
\frac1C(\frac12||w||^2 + C\sum_i\xi_i)
$$
与原始目标函数等价

合页损失函数图形如下，横轴为函数间隔，纵轴为损失，图中还画出来 0-1 损失函数，可以将它认为是二分类问题的真正的损失函数，合页损失函数为其上界，由于 0-1 损失不可导，直接优化比较困难，可以认为线性支持向量机是优化由 0-1 损失函数的上界构成的目标函数，这时的上界目标函数又称为代理损失函数（surrogate）

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210424124616288.png" alt="image-20210424124616288" style="zoom:50%;" />

虚线表示感知机的损失函数 $[y_i(wx_i + b)]_+$，当样本被正确分类时，损失为 0，否则为 $-y_i(wx_i + b)$，而合页损失函数不仅要正确分类，而且要确信度足够高损失才会为 0，也就是说合页损失函数对学习有更高的要求

# 非线性支持向量机和核技巧

## 核技巧

设原空间为 $X,x=(x^1,x^2)^T$，新空间为 $Z,z=(z^1,z^2)T$，定义从原空间到新空间的映射
$$
z = \phi(x) = ((x^1)^2,(x^2)^2)^T
$$
经过变换，原空间的点相应地变为新空间的点，原空间的椭圆
$$
w_1((x^1))^2 + w_2((x^2))^2 + b = 0
$$
变为新空间的直线
$$
w_1z^1 + w_2z^2 + b = 0
$$
在新空间里，直线可以将变换后的正负实例点正确分开，这样，原空间的非线性可分问题就变为了新空间的线性可分问题

核技巧就是这样：使用一个变换将原空间的数据映射到新空间，然后在新空间里用线性分类学习方法从训练数据中学习分类模型。

**核函数的定义**

设 X 为输入空间，H 为特征空间，如果存在一个 X 到 H 的映射
$$
\phi(x) = X\rightarrow H
$$
使得对所有 $x,z\in X$，函数 K(x,z) 满足条件
$$
K(x,z) = \phi(x)\cdot\phi(z)
$$
则称 K(x,z) 为核函数，$\phi(x)$ 为映射函数，$\phi(x)\cdot\phi(z)$ 为内积

核技巧的想法是：在学习和预测中只定义核函数 K(x,z)，而不显式的定义映射函数 $\phi$，通常直接计算 K(x,z) 更容易，

**核技巧在支持向量机中的应用**

将对偶问题的目标函数中的 $x_i\cdot x_j$ 用核函数 $K(x_i,x_j)=\phi(x_i)\phi(x_j)$ 代替，目标函数则变为了
$$
\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_jK(x_i,x_j) -\sum_i\alpha_i
$$
分类决策函数成为了
$$
f(x) = sign(\sum_i\alpha_i^*y_iK(x_i,x) + b^*)
$$

## 常用核函数

1. 多项式核函数
   $$
   K(x,z) = (x\cdot z + 1)^p
   $$
   支持向量机为 p 次多项式分类器，分类决策函数为

2. 高斯核函数
   $$
   K(x,z) = exp(-\frac{||x-z||^2}{2\sigma^2})
   $$
   支持向量机为高斯径向基函数分类器

## 非线性支持向量分类机

**定义** 从非线性分类训练集，通过核函数和软间隔最大化，或凸二次规划，学习得到的分类决策函数
$$
f(x) =sign(\sum_i\alpha_i^*y_iK(x\cdot x_i) + b^*)
$$
称为非线性支持向量机

**学习算法**

（1）选取适当的核函数 K(x,z) 和参数 C，构造并求解最优化问题，
$$
min_{\alpha}\ \sum_i\sum_j\alpha_i\alpha_jy_iy_jK(x_i\cdot x_j) -\sum_i\alpha_i\\
s.t.\quad \sum_i\alpha_iy_i = 0\\
0\leq\alpha_i\leq C
$$
​	求得最优解 $\alpha^*=(\alpha_1^*,\alpha_2^*\cdots)^T$

（2）计算 $w^*=\sum_i\alpha_i^*y_i K(x_i,x_j)$，选取 $0\leq\alpha_i^*\leq C$，计算
$$
b^* = y_j - \sum_i\alpha_iy_iK(x_i\cdot x_j)
$$
（3）构造决策函数
$$
f(x) = sign(\sum_i\alpha_i^*y_iK(x\cdot x_i) + b^*)
$$

# 序列最小最优化问题

支持向量机的学习问题可以形式化为求解凸二次规划问题，这样的问题具有全局最优解，并且许多算法可以用于求解，但当样本容量大时，会变得很低效，一致无法使用，下面讲述序列最小最优化（sequential minimal optimization）SMO 算法

SMO 算法要解以下凸二次规划的对偶问题：
$$
min_{\alpha}\ \frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_jK(x_i,x) - \sum_i\alpha_i\tag{7.98}\\
s.t.\quad \sum_i\alpha_iy_i=0\\
0\leq\alpha_i\leq C
$$
变量为拉格朗日乘子

整个 SMO 算法包括两部分：求解两个变量二次规划的解析方法和选择变量的启发式方法

## 两个变量二次规划的求解方法

假设选择的两个变量是 $\alpha_1,\alpha_2$，其他变量固定，则最优化问题 7.98 的子问题可写为
$$
min_{\alpha_1,\alpha_2}\ W(\alpha_1,\alpha_2) = \frac12K_{11}\alpha_1^2 + \frac12K_{22}\alpha_2^2 + y_1y_2K_{12}\alpha_1\alpha_2\\ - \alpha_1 - \alpha_2 + y_1\alpha_1\sum_{i=3}\alpha_iy_iK_{i1} + y_2\alpha_2\sum_{i=3}\alpha_iy_iK_{i2}\tag{7.101}\\
s.t.\quad \alpha_1y_1 + \alpha_2y_2 = -\sum_{i=3}\alpha_iy_i = \epsilon\\
0\leq\alpha_i\leq C
$$
$K_{ij}=K(x_i,x_j)，\epsilon$ 为常数，目标函数式 7.101 省略了不含 $\alpha_1,\alpha_2$ 的项

由于只有两个变量，约束可用二维空间中的图形表示

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210424172723176.png" alt="image-20210424172723176" style="zoom:50%;" />

不等式约束使得 $\alpha_1,\alpha_2$ 在盒子 [0,C]*[0,C] 的盒子内，等式约束使得他们在平行于盒子对角线的直线上，因此要求的是目标函数在平行与对角线的线段上的最优值，这使得两个变量的最优化问题称为实质上的单变量的最优化问题，不妨考虑 $\alpha_2$ 的最优化问题

假设 7.101 的初始可行解为 $\alpha_1^o,\alpha_2^o$，最优解为 $\alpha_1^n,\alpha_2^n$，并且假设在沿着约束方向未经剪辑时 $\alpha_2$ 的最优解为 $\alpha_2^{n,u}$

由于 $\alpha_2^n$ 满足不等式约束，所有取值范围是
$$
L\leq \alpha_2^n\leq H
$$
L,H 是 $\alpha_2^n$ 所在的线段端点的界，如果 $y_1\neq y_2$
$$
L=max(0,\alpha_2^o-\alpha_1^o)\ 
H=min(C,C+\alpha_2^o-\alpha_1^o)
$$
如果 $y_1 = y_2$
$$
L=max(0,\alpha_2^o+\alpha_1^o-C)\ H=min(C,\alpha_2^o+\alpha_1^o)
$$
下面首先求 $\alpha_2^{n,u}$，然后求 $\alpha_2^n$，记
$$
g(x) = \sum_i\alpha_iy_iK(x_i,x) + b\tag{7.104}
$$
令
$$
E_i = g(x_i)-y_i =(\sum_j\alpha_jy_jK(x_j,x_i)+b) - y_i,i=1,2\tag{7.105}
$$
当 i=1,2 时，$E_i$ 为函数 g(x) 对 $x_i$ 的预测值与真实输出 $y_i$ 之差

**定理** 最优化问题 7.101 沿着约束方向未经剪辑的解是                                                  
$$
\alpha_2^{n,u}=\alpha_2^o + \frac{y_2(E_1 - E_2)}{\eta}\tag{7.106}\\
\eta = K_{11} + K_{22} - 2K_{12} = (\Phi(x_1) - \Phi(x_2))^2
$$
经过剪辑后 
$$
\alpha_2^n = \begin{cases}
H,\quad\alpha_2^{n,u} > H\\
\alpha_2^{n,u},\quad L\leq\alpha_2^{u.n}\leq H\\
L,\quad \alpha_2^{n,u} < L
\end{cases}\\
\rightarrow \alpha_1^n = \alpha_1^o + y_1y_2(\alpha_2^o-\alpha_2^n)
$$

## 变量的选择方法

SMO 算法在每个子问题中选择两个变量优化，其中至少有一个是违反 KKT 条件的

1. 第一个变量的选择

   选择第一个变量的过程为外层循环，外层循环在训练样本中选取违反 KKT 条件最严重的样本点，然后将其对应的变量作为第一个变量，具体地检验样本点 $(x_i,y_i)$ 是否满足 KKT 条件，即
   $$
   \alpha_i=0\Leftrightarrow y_ig(x)\geq 1\\
   0<\alpha_i<C\Leftrightarrow y_ig(x)= 1\\
   \alpha_i=C\Leftrightarrow y_ig(x)\leq 1\tag{7.111}
   $$
   该检验是在 $\epsilon$ 范围内实现的，检验过程中，首先遍历所有满足 $0<\alpha_i< C$ 的样本，即支持向量。检验它们是否满足 KKT 条件，如果支持向量都满足，则遍历整个数据集

2. 第二个变量的选择

   选择第二个变量的过程称为内层循环，第二个变量的选择标准是希望 $\alpha_2$ 的变化足够大

   由 7.106 可知，$\alpha_2^n$ 是依赖于 $|E_1 - E_2|$ 的，为了加快计算速度，一种简单的方法是选择 $|E_1-E_2|$ 最大的 $\alpha_2$，因为 $\alpha_1$ 已定，那 $E_1$ 就是已知的，如果 $E_1>0$，选择最小的 $E_i$，如果 $E_1<0$，选择最大的 $E_i$，为了节省时间，可将所有的 $E_i$ 保存到列表中

   特殊情况下，如果通过以上方法找到的 $\alpha_2$ 没法儿使得目标函数有足够大的下降，那么采用下启发式规则选择 $\alpha_2$，依次将支持向量对应的变量作为 $\alpha_2$，直到目标函数有足够的下降，若找不到合适的，则遍历数据集，若仍找不到，则舍弃之前的 $\alpha_1$，通过外层循环寻找另外的 $\alpha_1$

3. 计算阈值 b 和差值 $E_i$

   每次完成两个变量的优化后，都要重新计算阈值 b，当 $0<\alpha_1^n<C$ 时，有 KKT 条件 7.112 可知
   $$
   \sum_i\alpha_iy_iK_{i1} + b=y_1\\
   b = y_1 - \sum_{i=3}\alpha_iy_iK_{i1} - \alpha_1^ny_1K_{11} - \alpha_2^ny_2K_{21}\tag{7.114}\\
   E_1 = \sum_{i=3}\alpha_iy_iK_{i1}+\alpha_1^oy_1K_{11}+\alpha_2^oy_2K_{21} + b^o -y_1
   $$
   式 7.114 的前两项可写成
   $$
   y_1 - \sum_{i=3}\alpha_iy_iK_{i1} = -E_1+\alpha_1^oy_1K_{11}+\alpha_2^oy_2K_{21} + b^o\\
   \rightarrow b_1^n = -E_1 -y_1K_{11}(\alpha_1^n-\alpha_1^o) - y_2K_{21}(\alpha_2^n-\alpha_2^o)+b^o
   $$
   如果 $0<\alpha_2^n<C$，有
   $$
   b_2^n = -E_2 -y_2K_{22}(\alpha_2^n-\alpha_2^o) - y_1K_{12}(\alpha_1^n-\alpha_1^o)+b^o
   $$
   如果 $\alpha_1^n,\alpha_2^n$ 同时满足不等式，则 $b_1^n = b_2^n$，如果值为 0 或 C，那么 $b_1^n,b_2^n$ 及它们之间的数都是符合 KKT 条件的阈值，这时选择它们的中点作为 $b^n$

   更新 $E_i$ 的值，并保存到列表中，
   $$
   E_i^n = \sum_Sy_j\alpha_jK(x_i,x_j) + b^n - y_i
   $$
   S 为所有支持向量的集合

## SMO 算法

（1）取初值 $\alpha^0 = 0,k=0$

（2）选取优化变量 $\alpha_1^k,\alpha_2^k$，解析求解两个变量的最优化问题 7.101，求得最优解 $\alpha_1^{k+1},\alpha_2^{k+1}$，更新 $\alpha$ 为 $\alpha^{k+1}$

（3）若在精度 $\epsilon$ 范围内满足停机条件
$$
\sum_i\alpha_iy_i = 0\\
0\leq\alpha_i\leq C\\
y_i\cdot g(x) = \begin{cases}
\geq 1,\qquad\{x_i|\alpha_i=0\}\\
=1,\qquad\{x_i|0<\alpha_i<C\}\\
\leq 1,\qquad\{x_i|\alpha_i=C\}
\end{cases}\\
g(x) = \sum_i\alpha_iy_iK(x_i,x_j) + b
$$
则转（4），否则令 k=k+1，转（2）

（4）取 $\hat\alpha=\alpha^{k+1}$































