条件随机场（CRF）是给定一组输入随机变量条件下另一组输出随机变量的条件概率分布模型，其特点是假设输出随机变量构成马尔可夫随机场。条件随机场可以用于不同的预测问题，本书仅论及它在标注问题的应用。因此主要讲述线性链（linear chain）条件随机场，这时，问题变成了有输入序列对输出序列预测的判别模型，形式为对数线性模型，学习方法通常是极大似然估计或正则化的极大似然估计

# 概率无向图模型

概率无向图模型（probabilistic undirected graphical model），又称马尔可夫随机场（Markov random field）是一个可以有无向图表示的联合概率分布

## 模型定义

图（graph）是由结点（node）及连接结点的边（edge）组成的集合，结点和边分别记作 $v$ 和 $e$，结点和边的集合分别记作 V 和 E，图记作 $G=(V,E)$。无向图是指边没有方向的图

概率图模型是由图表示的概率分别。设有联合概率分布 $P(Y)$，Y 是一组随机变量，由无向图 G 表示概率分布 $P(Y)$，即在 G 中，结点 $v$ 表示一个随机变量 $Y_v$，$Y=(Y_v)_{v\in V}$；边 $e\in E$ 表示随机变量之间的概率依赖关系

给定一个联合概率分布 $P(Y)$ 和表示它的无向图 G，首先定义无向图表示的随机变量之间的成对马尔可夫性（pairwise Markov property）、局部马尔可夫性（local）和全局马尔可夫性（global）

成对马尔可夫性：设 u 和 v 是无向图 G 中任意两个没有边连接的结点，结点 u 和 v 分别对应随机变量 $Y_u,Y_v$，其他所有结点为 O，对应的随机变量组是 $Y_O$，成对马尔可夫性是指给定随机变量组 $Y_O$ 的条件下随机变量 $Y_u,Y_v$ 是条件独立的，即
$$
P(Y_u,Y_v|Y_O) = P(Y_u|Y_O)P(Y_v|Y_O)
$$
局部：设 v 是 G 中任意一个节点， W 是与 v 有边连接的所有结点，O 是 v,W 以外的其他所有结点，局部马尔可夫性是指在给定随机变量组 $Y_W$ 的条件下随机变量 $Y_v$ 与随机变量组 $Y_O$ 是独立的
$$
P(Y_v,Y_O|Y_W) = P(Y_v|Y_W)P(Y_O|Y_W)
$$
在 $P(Y_O|Y_W)>0$ 时，有
$$
P(Y_v|Y_W) = P(Y_v|Y_W,Y_O)
$$
下图表示局部马尔可夫性

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210517113949525.png" alt="image-20210517113949525" style="zoom:50%;" />

全局：设结点集合 A，B 是 G 中被结点集合 C 分开的任意结点集合，如下图，全局马尔可夫性是指给定随机变量组 $Y_C$ 条件下随机变量组 $Y_A$ 和 $Y_B$ 是条件独立的
$$
P(Y_A,Y_B|Y_C) = P(Y_A|Y_C)P(Y_B|Y_C)
$$
<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210517114316708.png" alt="image-20210517114316708" style="zoom:50%;" />

**定义 1 概率无向图模型** 设有联合概率分布 P(Y)，由无向图 G 表示，在 G 中，结点表示随机变量，边表示随机变量之间的依赖关系。如果联合概率分布 P(Y) 满足成对、局部或全局马尔可夫性，就称此联合概率分布为概率无向图模型，或马尔可夫随机场

实际上，我们更关心的是如何求其联合概率分布，对给定的概率无向图模型，我们希望将整体的联合概率写成若干子联合概率的乘积的形式，也就是将联合概率进行因子分解，这一便于模型的学习与计算。

## 因子分解

**定义 2 团与最大团** G 中任何两个结点均有边连接的结点子集称为团（clique），若 C 是 G 的一个团，并且不能再加进任何一个 G 的结点使其成为一个更大的团，则称此 C 为最大团

下图表示由 4 个结点组成的无向图，成对结点组成的团有 5 个。两个最大团：$\{Y_1,Y_2,Y_3\},\{Y_2,Y_3,Y_4\}$。而 $\{Y_1,Y_2,Y_3,Y_4\}$ 不是一个团，因为 $Y_1,Y_4$ 没有连接

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210517120438411.png" alt="image-20210517120438411" style="zoom:67%;" />

将概率无向图模型的联合概率分布表示为其最大团上的随机变量的函数的乘积形式的操作，成为模型的因子分解

给定概率无向图模型，设其无向图为 G，C 为 G 上的最大团，$Y_C$ 表示 C 对应的随机变量，那么概率无向图模型的联合概率分布 $P(Y)$ 可写作图中所有最大团 C 上的函数 $\Psi_C(Y_C)$ 的乘积形式，
$$
P(Y) = \frac1Z\prod_C\Psi_C(Y_C)\\
Z = \sum_Y\frac1Z\prod_C\Psi_C(Y_C)
$$
Z 保证 P(Y) 构成一个概率分布。函数 $\Psi_C(Y_C)$ 称为势函数（potential）。这里要求 $\Psi_C(Y_C)$ 是严格正的，通常定义为指数函数
$$
\Psi_C(Y_C) = -exp(-E(Y_C))
$$
概率无向图模型的因子分解由下述定理来保证

**定理 1（Hammersley-Clifford 定理）** 概率无向图模型的联合概率分布 P(Y) 可表示如下
$$
P(Y) = \frac1Z\prod_C\Psi_C(Y_C)\\
Z = \sum_Y\frac1Z\prod_C\Psi_C(Y_C)
$$
C 是无向图的最大团，$Y_C$ 是 C 的结点对应的随机变量，$\Psi_C(Y_C)$ 是 C 上定义的严格正函数，乘积是再无向图所有的最大团上进行的

# 条件随机场的定义与形式

## 定义

条件随机场（conditional random field）是给定随机变量 X 条件下，随机变量 Y 的马尔可夫随机场。这里主要介绍定义在线性链上的特殊的条件随机场，称为线性链条件随机场（linear chain）。线性链条件随机场可以用于标注等问题，这时，再条件概率模型 P(Y|X) 中，Y 是输出变量，表示标记序列，X 是输入变量，表示需要标注的观测序列。也把标记序列称为状态序列。学习时，利用训练数据集通过极大似然估计或正则化的极大似然估计得到条件概率模型 $\hat P(Y|X)$；预测时，对应给定的输入序列 x，求出条件概率 $\hat P(y|x)$ 最大的输出序列 $\hat y$

**定义 3 条件随机场** 设 X 与 Y 是随机变量，P(Y|X) 是在给定 X 的条件下 Y 的条件概率分布。若随机变量 Y 构成一个由 G 表示的马尔可夫随机场
$$
P(Y_v|X,Y_w,w\neq v) = P(Y_v|X,Y_w,w-v)
$$
对任意结点 v 成立，则条件概率分布 P(Y|X) 为条件随机场。式中 w-v 表示在 G 中与结点 v 有边连接的所有结点 w，$w\neq v$ 表示结点 v 以外的所有结点，$Y_v,Y_u,Y_w$ 为随机变量

定义中并没有要求 X 和 Y 有相同的结构，现实中，一般假设 X 和 Y 有相同的图结构，此处主要考虑无向图如下所示的线性链的情况
$$
G=(V=\{1,2,\cdots,n\},E=\{(i,i+1)\}),i=1,2,\cdots,n-1
$$
$X = (X_1,X_2,\cdots,X_n),Y=(Y_1,Y_2,\cdots,Y_n)$，最大团是相邻两个结点的集合

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210517124430851.png" alt="image-20210517124430851" style="zoom:67%;" />

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210517124504476.png" alt="image-20210517124504476" style="zoom:67%;" />

**定义 4 线性链条件随机场** X，Y 均为线性链表示的随机变量序列，若在给定随机变量序列 X 的条件下，随机变量序列 Y 的条件概率分布 P(Y|X) 构成条件随机场，即满足马尔可夫性
$$
P(Y_i|X,Y_1,Y_2,\cdots,Y_n) = P(Y_i|X,Y_{i-1},Y_{i+1}),i=1,2,\cdots,n
$$
则称 P(Y|X) 为线性链条件随机场，在标注问题中，X 表示输入观测序列，Y 表示对应的输出标记序列或状态序列

## 参数化形式

**定理 2（线性链条件随机场的参数化形式）** 设 P(Y|X) 为线性链条件随机场，则在随机变量 X 取值为 x 的条件下，随机变量 Y 取值为 y 的条件概率具有如下形式
$$
P(y|x) = \frac1{Z(x)} exp(\sum_{i,k}\lambda_kt_k(y_{i-1},y_i,x,i) + \sum_{i,l}\mu_is_l(y_i,x,i))
$$
$t_k,s_l$ 是特征函数，$\lambda_k,\mu_l$ 为对应的权值。Z(x) 为规范化因子，求和是在所有可能的输出序列上进行的

上式为线性链条件随机场模型的基本形式，表示给定输入序列 x，对输出序列 y 预测的条件概率。$t_k$ 是定义在边上的特征函数，称为转移特征，依赖于当前和前一个位置，$s_l$ 是定义在结点上的特征函数，称为状态特征，依赖于当前位置。这两者都依赖于位置，是局部特征函数。通常，这两个特征函数取值为 1 或 0；当满足特征条件是取值为 1，否则为 0，

线性链条件随机场也是对数线性模型（log linear model）

例 设有一标注问题：输入观测序列为 $X = (X_1,X_2,X_3)$，输出标记序列为 $Y=(Y_1,Y_2,Y_3)$，Y 取值于 {1，2}

$t_k,s_l,\lambda_k,\mu_l$  取值如下
$$
\begin{align}&t_1 = t_1(y_{i-1}=1,y_i=2,x,i),i=2,3,\lambda_1=1\\
&t_1 = \begin{cases}
1,\qquad y_{i-1}=1,y_i=2,x,i(i\in [2,3])\\
0,\qquad 其他
\end{cases}\\
&t_2 = t_2(y_1=1,y_2=1,x,2)\quad \lambda_2=0.5\\
&t_3 = t_3(y_2=2,y_3=1,x,3)\quad \lambda_3=1\\
&t_4 = t_4(y_1=2,y_2=1,x,2)\quad \lambda_4 = 1\\
&t_5 = t_5(y_2=2,y_3=2,x,3)\quad \lambda_5 = 0.2\\
&s_1 = s_1(y_1=1,x,1)\quad \mu_1=1\\
&s_2 = s_2(y_i=2,x,i),i\in[1,2]\quad \mu_2 = 0.5\\
&s_3 = s_3(y_i=1,x,i),i\in[2,3]\quad \mu_3=0.8\\
&s_4 = s_4(y_3=2,x,3),\quad \mu_4=0.5
\end{align}
$$
对给定的观测序列 x，求标记序列为 $y=(y_1,y_2,y_3)=(1,2,2)$ 的非规范化条件概率（没有除以规范化因子的条件概率）

解：模型为
$$
\begin{align}P(y|x) &= exp[\sum_k\lambda_k\sum_{i=2}^3t_k(y_{i-1},y_i,x,i) +\sum_l\mu_l\sum_is_l(y_i,x,i)]\\
&= exp[\lambda_1t_1(y_1=1,y_2=2,x,2) + \lambda_5t_5(y_2=2,y_3=2,x,3) + \\
&\qquad\quad\mu_1s_1(y_1=1,x,1) +\mu_2s_2(y_2=2,x,2) + \mu_4s_4(y_3=2,x,3)]\\
&=exp(3.2)
\end{align}
$$

## 简化形式

条件随机场的参数化形式中同一特征在各个位置都有定义，可以对同一个特征在各个位置求和，将局部特征函数转换为一个全局特征函数，这样就可以将条件随机场写成权值向量和特征向量的内积形式

设有 $K_1$ 个转移特征，$K_2$ 个状态特征，$K = K_1+K_2$
$$
f_k(y_{i-1},y_i,x,i)\begin{cases}
t_k(y_{i-1},y_i,x,i),\quad k\in[1,K_1]\\
s_l(y_i,x,i),\quad k=K_1+l;l\in[1,K_2]
\end{cases}
$$
对转移与状态特征在各个位置 i 求和，记作
$$
f_k(y,x) = \sum_if_k(y_{i-1},y_i,x,i),k\in[1,K]
$$
$w_k$ 表示特征 $f_k$ 的权值
$$
w_k = \begin{cases}
\lambda_k,\quad k\in[1,K_1]\\
\mu_l,\quad k=K_1+l;l\in[1,K_2]
\end{cases}
$$
于是条件随机场可表示为
$$
P(y|x) = \frac1{Z(x)}exp\sum_kw_kf_k(y,x)\tag{1}\\
Z(z) = \sum_yexp\sum_kw_kf_k(y,x)
$$
若以 w 表示权值向量
$$
w = (w_1,w_2,\cdots,w_K)^T
$$
以 F(y,x) 表示全局特征向量
$$
F(y,x) = (f_1(y,x),f_2(y,x),\cdots,f_K(y,x))^T
$$
则条件随机场可写成向量 w 与 F(y,x) 的内积的形式
$$
P_w(y|x) = \frac1{Z_w(x)}exp(w\cdot F(y,x))\\
Z_w(x) = \sum_yexp(w\cdot F(y,x))
$$

## 矩阵形式

对公式 1 表示的线性链条件随机场，引进特殊的起点和终点状态标记 $y_0=start,y_{n+1}=stop$，这时 $P_w(y|x)$ 可以通过矩阵形式表示

对观测序列 x 的每一个位置 $i\in[1,n+1]$ 定义一个 m 阶矩阵（m 是标记 $y_i$ 取值的个数）
$$
M_i(x) = [M_i(y_{i-1},y_i|x)]\\
M_i(y_{i-1},y_i|x) = exp(W_i(y_{i-1},y_i|x))\\
W_i(y_{i-1},y_i|x) = \sum_iw_kf_k(y_{i-1},y_i,x,i)
$$
给定观测序列 x，标记序列 y 的非规范化概率可以通过 n+1 个矩阵的乘积 $\prod_i M_i(y_{i-1},y_i|x)$ 表示，于是
$$
P_w(y|x) = \frac1{Z_w(x)}\prod_iM_i(y_{i-1},y_i|x)\\
Z_w(x) = (M_1(x)M_2(x)\cdots M_{n+1}(x))_{start,stop}\tag{2}
$$
规范化因子是以 start 为起点 stop 为终点通过状态的所有路径 $y_1y_2\cdots y_n$ 的非规范化概率 $\prod_iM_i(y_{i-1},y_i|x)$ 之和

例 给定一个由下图所示的线性链条件随机场，观测序列 x，状态序列 y，i=1,2,3,n=3，标记 $y_i\in\{1,2\}$，假设 $y_0=start=1,y_4=stop=1$，各个位置的随机矩阵分别是
$$
M_1(x)\left[\begin{matrix}a_{01}&a_{02}\\0&0 \end{matrix}\right],
M_2(x)\left[\begin{matrix}b_{11}&b_{12}\\b_{21}&b_{22}
\end{matrix}\right]\\
M_3(x)\left[\begin{matrix}c_{11}&c_{12}\\c_{21}&c_{22}
\end{matrix}\right],
M_4(x)\left[\begin{matrix}1&0\\1&0
\end{matrix}\right]
$$
试求状态序列 y 以 start 为起点 stop 为终点所有路径的非规范化概率及规范化因子

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210517170241007.png" alt="image-20210517170241007" style="zoom:67%;" />

解 从 start 到 stop 对应于 y=(1,1,1)，y=(1,1,2)，...，y=(2,2,2) 各路径的非规范化概率分别是
$$
a_{01}b_{11}c_{11},a_{01}b_{11}c_{12},a_{01}b_{12}c_{21},a_{01}b_{12}c_{22}\\
a_{02}b_{21}c_{11},a_{02}b_{21}c_{12},a_{02}b_{22}c_{21},a_{02}b_{22}c_{22}
$$
按公式 2 求规范化因子，通过计算矩阵乘积 $M_1(x)M_2(x)M_3(x)M_4(x)$ 可知，第一行第一列的元素为
$$
a_{01}b_{11}c_{11}+a_{01}b_{11}c_{12}+a_{01}b_{12}c_{21}+a_{01}b_{12}c_{22}\\
+a_{02}b_{21}c_{11}+a_{02}b_{21}c_{12}+a_{02}b_{22}c_{21}+a_{02}b_{22}c_{22}
$$
即为以 start 为起点 stop 为终点所有路径的非规范化概率之和

# 概率计算问题

条件随机场的概率计算问题是给定条件随机场 P(Y|X)，输入序列 x 和输出序列 y，计算条件概率 $P(Y_i=y_i|x),P(Y_{i-1}=y_{i-1},Y_i=y_i|x)$ 以及相应的数学期望的问题

## 前向-后向算法

对每个指标 i=0,1,...,n+1，定义前向向量 $\alpha_i(x)$
$$
\alpha_0(y|x) = \begin{cases}
1,\quad y=start\\
0,\quad 其他
\end{cases}
$$
递推公式为
$$
\alpha_i^T(y_i|x) = \alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x),i\in[1,n+1]\\
等价于\qquad\alpha_i^T(x) = \alpha_{i-1}^T(x)M_i(x)
$$
$\alpha_i(y_i|x)$ 表示在位置 i 的标记是 $y_i$ 并且到位置 i 的前部分标记序列的非规范化概率，$y_i$ 的取值有 m 个，$\alpha_i(x)$ 是 m 维列向量

同样，对每个指标 $i\in[0,n+1]$，定义后向向量 $\beta_i(x)$
$$
\beta_{n+1}(y_{n+1}|x) = \begin{cases}
1,\quad y_{n+1}=stop\\
0,\quad else
\end{cases}\\
\beta_i(y_i|x) = M_i(y_i,y_{i+1}|x)\beta_{i-1}(y_{i+1}|x)\\
等价于\quad \beta_i(x) = M_{i+1}(x)\beta_{i+1}(x)
$$
$\beta_i(y_i|x)$ 表示在位置 i 的标记为 $y_i$ 并且从 i+1 到 n 的后部分标记序列的非规范化概率

由前向-后向向量定义可得
$$
Z(x) = \alpha_n^T(x)\cdot \vec1=\vec1^T\cdot \beta_i(x)
$$
$\vec1$ 是元素均为 1 的 m 维列向量

## 概率计算

按照前向-后向向量的定义，可以得到
$$
P(Y_i=y_i|x) = \frac{\alpha_i^T(y_i|x)\beta_i(y_i|x)}{Z(x)}\\
P(Y_{i-1}=y_{i-1},Y_i=y_i|x)=\frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}\\
Z(x) = \alpha_n^T(x)\cdot \vec1
$$

## 期望值的计算

特征函数 $f_k$ 关于条件分布 P(Y|X) 的数学期望是
$$
\begin{align}E_{P(Y|X)}[f_k]&=\sum_yP(y|x)f_k(y,x)\\
&=\sum_{i\in[i,n+1]}\sum_{y_{i-1}y_i}f_k(y_{i-1},y_i,x,i)\frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}
\end{align}\\
Z(x) = \alpha_n^T(x)\cdot\vec1,k\in[1,K]
$$
假设经验分布为 $\tilde P(X)$，特征函数 $f_k$ 关于联合分布 $P(X,Y)$ 的数学期望是
$$
\begin{align}E_{P(X,Y)}[f_k] &= \sum_{x,y}P(x,y)\sum_{i\in[1,n+1]}f_k(y_{i-1},y_i,x,i)\\
&=\sum_x\tilde P(x)\sum_y\sum_yP(y|x)\sum_{i\in[1,n+1]}f_k(y_{i-1},y_i,x,i)\\
&=\sum_x\tilde P(x)\sum_{i\in[i,n+1]}\sum_{y_{i-1}y_i}f_k(y_{i-1},y_i,x,i)\frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}
\end{align}
$$
上述两个式子是特征函数数学期望的一般计算公式。对与转移特征 $t_k(y_{i-1},y_i,x,i),k\in[1,K_1]$，可将式中的 $f_k$ 换成 $t_k$；对于状态特征，可将 $f_k$ 换成 $s_l,k=K_1+l,l\in[1,K_2]$，

以上，对应给定的观测序列 x 和标记序列 y，可以通过一次前向扫描计算 $\alpha_i,Z(x)$，通过一次后向扫描计算 $\beta_i$，从而计算所有的概率和特征的期望

# 学习算法

已知训练数据集，可得经验概率分布 $\tilde P(X,Y)$，可以通过极大化训练数据的对数似然函数来求模型参数

训练数据的对数似然函数为
$$
L(w) = L_{\tilde P}(P_w) = log\prod_{x,y}P_w(y|x)^{\tilde P(X,Y)}=\sum_{x,y}\tilde P(X,Y)logP_w(y|x)
$$
当 $P_w$ 是一个由公式 1 给出的条件随机场模型时，对数似然函数为
$$
\begin{align}L(w)&=\sum_{x,y}\tilde P(X,Y)logP_w(y|x)\\
&=\sum_{x,y}[\tilde P(X,Y)\sum_kw_kf_k(y,x)-\tilde P(X,Y)logZ_w(x)]\\
&=\sum_{j\in[1,N]}\sum_{k\in[1,K]}w_kf_k(y_j,x_j)-\sum_jlogZ_w(x_j)
\end{align}
$$
改进的迭代尺度法通过迭代的方法不断优化对数似然函数改变量的下界，达到极大化对数似然函数的目的。假设模型的当前参数向量为 $w=(w_1,w_2,\cdots,w_K)^T$，向量的增量为 $\delta=(\delta_1,\delta_2,\cdots,\delta_K)^T$，更新参数向量为 $w+\delta$，在每步迭代过程中，改进的迭代尺度法通过一次求解下面的两个式子，得到 $\delta$

关于特征转移 $t_k$ 的更新方程为：
$$
\begin{align}E_{\tilde P}[t_k] &= \sum_{x,y}\tilde P(X,Y)\sum_{i\in[1,n+1]}t_k(y_{i-1},y_i,x,i)\\
&=\sum_{x,y}\tilde P(x)P(y|x)\sum_it_k(y_{i-1},y_i,x,i)exp(\delta_kT(x,y))\tag{3}
\end{align}\\
k\in[1,K_1]
$$
关于状态特征 $s_l$ 的更新方程为
$$
\begin{align}E_{\tilde P}[s_l] &= \sum_{x,y}\tilde P(X,Y)\sum_is_l(y_i,x,i)\\
&=\sum_{x,y}\tilde P(x)P(y|x)\sum_is_l(y_i,x,i)exp(\delta_{K_1+l}T(x,y))\tag{4}
\end{align}\\
l\in[1,K_2]\\
T(x,y) = \sum_kf_k(y,x) = \sum_k\sum_if_k(y_{i-1},y_i,x,i)
$$
**算法 1**

输入：特征函数 $t_k,s_l$；经验分布 $\tilde P(x,y)$

输出：参数估计值 $\hat w$，模型 $P_{\hat w}$

（1）对所有 $k\in[1,K]$，取初值 $w_k=0$

（2）对每一 $k\in[1,K]$

​	（a）当 $k\in[1,K_1]$，令 $\delta_k$ 是方程
$$
E_{\tilde P}[t_k] = \sum_{x,y}\tilde P(x)P(y|x)\sum_it_k(y_{i-1},y_i,x,i)exp(\delta_kT(x,y)
$$
​	的解

​	当 $k\in[K_1,K]$，令 $\delta_{K_1+l}$ 是方程
$$
E_{\tilde P}[s_l] = \sum_{x,y}\tilde P(x)P(y|x)\sum_{i\in[1,n]}s_l(y_i,x,i)exp(\delta_{K_1+l}T(x,y))
$$
​	的解

​	（b）更新 $w_k$ 的值：$w_k\leftarrow w_k+\delta_k$

（3）如果不是所有 $w_k$ 都收敛，重复步骤（2）

​	在公式 3 和 4 中，T(x,y) 表示数据 (x,y) 中的特征总数，对不同的数据 (x,y) 可能取值不同，为了处理这个问题，定义松弛特征
$$
s(x,y)  = S-\sum_i\sum_kf_k(y_{i-1},y_i,x,i)
$$
S 为常数，选择足够大的常数 S 使得对训练数据集的所有数据 (x,y)，$s(x,y)\geq 0$ 成立。这时特征总数可取 S

由式 3，转移特征 $t_k$，$\delta_k$ 的更新方程为
$$
E_{\tilde P}[t_k] = \sum_{x,y}\tilde P(x)P(y|x)\sum_it_k(y_{i-1},y_i,x,i)exp(\delta_kS)\\
\delta_k=\frac1Slog\frac{E_{\tilde P[t_k]}}{E_{P[t_k]}}\\
E_{P[t_k]}=\sum_x\tilde P(x)\sum_{i\in[i,n+1]}\sum_{y_{i-1}y_i}t_k(y_{i-1},y_i,x,i)\frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}
$$
由式 4，状态特征 $s_l,\delta_k$ 的更新方程是
$$
E_{\tilde P}[s_l] = \sum_{x,y}\tilde P(x)P(y|x)\sum_is_l(y_i,x,i)exp(\delta_{K_1+l}S)\\
\delta_k=\frac1Slog\frac{E_{\tilde P[s_l]}}{E_{P[s_l]}}\\
E_P(s_l) = \sum_x\tilde P(x)\sum_{i\in[i,n]}\sum_{y_i}s_l(y_i,x,i)\frac{\alpha_{i}^T(y_{i}|x)\beta_i(y_i|x)}{Z(x)}
$$
以上算法称为算法 S。在算法 S 中需要使常数 S 足够大，这样一来，每步迭代的增量向量会变大，算法收敛会变慢，算法 T 试图解决这个问题，算法 T 对每个观测序列 x 计算其特征总数最大值 T(x)
$$
T(x) = max_yT(x,y)
$$
利用前向-后向递推公式，可得 T(x) = t

这时，$t_k$ 的参数更新为
$$
\begin{align}E_{\tilde P}[t_k] &= \sum_{x,y}\tilde P(x)P(y|x)\sum_it_k(y_{i-1},y_i,x,i)exp(\delta_kT(x))\\
 &= \sum_{x}\tilde P(x)\sum_yP(y|x)\sum_it_k(y_{i-1},y_i,x,i)exp(\delta_kT(x))\\
 &=\sum_x\tilde P(x)a_{k,t}exp(\delta_k\cdot t)\\
 &=\sum_{t\in[0,T_{max}]}a_{k,t}\beta_k^t
\end{align}
$$
$a$ 表示特征 $t_k$ 的期待值，$\delta_k=log\beta_k。\beta_k$ 是上述方程唯一的实根，从而求得相关的 $\delta_k$

同样 $s_l$ 的参数更新为
$$
\begin{align}E_{\tilde P}[s_l] &= \sum_{x,y}\tilde P(x)P(y|x)\sum_{i\in[1,n]}s_l(y_i,x,i)exp(\delta_{K_1+l}T(x))\\
&=\sum_{x}\tilde P(x)\sum_yP(y|x)\sum_{i\in[1,n]}s_l(y_i,x,i)exp(\delta_{K_1+l}T(x))\\
&=\sum_x\tilde P(x)b_{l,t}exp(\delta_k\cdot t)\\
&=\sum_{t\in[0,T_{max}]}b_{l,t}\gamma_l^t
\end{align}
$$
$b_{l,t}$ 是特征 $s_l$ 的期望值，$\delta_l = log\gamma_l。\gamma_l$ 是上述方程唯一的实根

## 拟牛顿法

对于条件随机场模型
$$
P_w(y|x) = \frac{exp(\sum_{i\in[1,n]}w_if_i(x,y))}{\sum_yexp(\sum_iw_if_i(x,y))}
$$
学习的优化目标函数是
$$
min_wf(w) = \sum_x\tilde P(x)log\sum_yexp(\sum_{i\in[1,n]}w_if_i(x,y))-\sum_{x,y}\tilde P(x,y)\sum_iw_if_i(x,y)
$$
梯度函数是
$$
g(w) = \sum_{x,y}\tilde P(x)P_w(y|x)f(x,y)-E_{\tilde P}(f)
$$
拟牛顿法的 BFGS 算法如下

**算法 2**

输入：特征函数 $f_n$；经验分布 $\tilde P(X,Y)$

输出：最优参数值 $\hat w$；最优模型 $P_{\hat w}(y|x)$

（1）选定初始点 $w^0$，取 $B_0$ 为正定对称矩阵，令 k=0

（2）计算 $g_k = g(w^k)$。若 $g_k=0$，则停止计算，否则转（3）

（3）由 $B_kp_k = -g_k$ 求出 $p_k$

（4）一维搜索：求 $\lambda_k$ 使得
$$
f(w^k + \lambda_kp_k) = min_{\lambda\geq0}f(w^k+\lambda p_k)
$$
（5）令 $w^{k+1}=w^k+\lambda_kp_k$

（6）计算 $g_{k+1} = g(w^{k+1})$，若 $g_k=0$，则停止计算，否则求出 $B_{k+1}$
$$
B_{k+1} = B_k + \frac{y_ky_k^T}{y_k^T\delta_k} - \frac{B_k\delta_k\delta_k^TB_k}{\delta_k^TB_k\delta_k}\\
y_k=g_{k+1}-g_k,\delta_k=w^{k+1}-w^k
$$
（7）令 k=k+1，转（3）

# 预测算法

条件随机场的预测问题是给定条件随机场 P(Y|X) 和输入序列（观测序列）x，求条件概率最大的输出序列（标记序列）$y^*$，即对观测序列进行标注，条件随机场的预测算法是维特比算法
$$
\begin{align}y^* &= argmax_y P_w(y|x)\\
&=argmax_y\frac{exp(w\cdot F(y,x))}{Z_w(x)}\\
&=argmax_yexp(w\cdot F(y,x))\\
&=argmax_y(w\cdot F(y,x))
\end{align}
$$
条件随机场的预测问题就变成了求非规范化概率最大的最优路径问题
$$
max_y(w\cdot F(y,x))
$$
这里的路径表示标记序列
$$
w = (w_1,w_2,\cdots,w_k)^T\\
F(y,x)=(f_1(y,x),f_2(y,x),\cdots,f_K(y,x))^T\\
f_k(y,x) = \sum_{i\in[1,n]}f_k(y_{i-1},y_i,x,i),k\in[1,K]
$$
这时只需要计算非规范化概率，而不必计算概率，可以大大提高效率。为了求解最优路径，上式可写为
$$
max_y\sum_{i\in[1,n]}w\cdot F_i(y_{i-1},y_2,x)\\
F_i(y_{i-1},y_i,x)=(f_1(y_{i-1},y_i,x,i),f_2(y_{i-1},y_i,x,i),\cdots,f_K(y_{i-1},y_i,x,i))
$$
是局部特征向量

**算法 3 维特比算法**

输入：模型特征向量 F(y,x) 和权值向量 w，观测序列 $x=(x_1,x_2,\cdots,x_n)$

输出：最优路径 $y^*=(y_1^*,y_2^*,\cdots,y_n^*)$

（1）初始化(m 表示所有观测取值的集合的长度)
$$
\delta_1(j) = w\cdot F_1(y_0=start,y_1=j,x),j\in[1,m]
$$
（2）递推，对 i=2,3,...,n
$$
\delta_i(l)=max_{j\in[1,m]}\{\delta_{i-1}(j)+w\cdot F_i(y_{i-1}=j,y_i=l,x)\},l\in[1,m]\\
\Psi_i(l)=argmax_{j\in[1,m]}\{\delta_{i-1}(j)+w\cdot F_i(y_{i-1}=j,y_i=l,x)\},l\in[1,m]
$$
（3）i = n，终止
$$
max_y(w\cdot F(y,x)) = max_{j\in[1,m]}\delta_n(j)\\
y_n^*=argmax_{j\in[1,m]}\delta_n(j)
$$
（4）返回路径
$$
y_i^* = \Psi_{i+1}(y_{i+1}^*),i\in[n-1,n-2,\cdots,1]
$$




