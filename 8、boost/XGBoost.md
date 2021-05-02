XGBoost 又叫 ExtremeGBoost 本质上还是 gbdt，只是将 gbdt 的速度和效率发挥到了极致

## 定义

假如要预测一家人对电子游戏的喜好程度，考虑年龄，年轻人更可能喜欢电子游戏，考虑性别，男性更喜欢电子游戏，故先根据年龄大小区分小孩，大人，然后通过性别区分男女，逐一给各人在电子游戏喜好程度上打分，如下

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\aHR0cDovL3d3dy50ZW5zb3JmbG93bmV3cy5jb20vd3AtY29udGVudC91cGxvYWRzLzIwMTgvMDcvNS0yLnBuZw" alt="img" style="zoom:67%;" />

如此，训练出两棵树，与 gbdt 类似，两棵树的结果累加起来就是最终的结论，所以小男孩的预测分数就是：2 + 0.9 = 2.9。爷爷的预测分数为：-1 - 0.9 = -1.9

XGBoost 核心思想：

1. 不断地添加树，不断地进行特征分裂来生成一棵树，每次添加一棵树，其实就是学习一个新函数，去拟合上次预测的残差
   $$
   \hat y = \phi(x_i) = \sum_m f_m(x_i)\\
   where F=\{f(x)=w_{q(x)}\}(q:R^m\rightarrow T,w\in R^T)
   $$
   $w_{q(x)}$ 为叶子节点 q 的分数，F 对应了所以 M 棵树的集合，f(x) 表示其中一棵树

2. 当我们训练完得到 M 棵树，预测一个样本的分数，其实就是根据这个样本的特征，在每棵树中会落到对应的一个叶子节点，每个叶子节点就对应一个分数

3. 最好将每棵树对应的分数加起来就是该样本的预测值

到目前为止，XGBoost 貌似与 gbdt 是一样的

事实上，如果不考虑工程实现、解决问题的差异，xgboost 与 gbdt 比较大的不同就是目标函数的定义，如下
## 目标函数

我们的目标是要使得树群的预测值 $\hat y_i$ 尽可能接近真实值 $y_i$，而且有尽量大的泛化能力

从数学角度看这是一个泛函最优化问题，故把目标函数简化为：
$$
L(\phi) = \sum_iL(y_i - \hat y_i) + \sum_m\Omega(f_m)
$$
这个函数有损失函数和正则化项两部分组成。

就上式而言 $\hat y_i$ 是整个累加模型的输出，正则化项 $\sum_m\Omega(f_m)$ 是表示树的复杂度的函数，值越小，复杂度越低，泛化能力越强
$$
\Omega(f) = \gamma T + \frac12\lambda||w||^2
$$
T 表示叶结点个数，w 表示叶结点的分数，直观上看，目标要求预测误差尽量小，且叶子节点 T 尽量少（$\gamma$ 控制叶子节点个数），节点数值 w 尽量不过大，防止过拟合

> 一般的目标函数都包含下面两项
> $$
> J(w) = L(w) + \Omega(w)
> $$
> 第一项为损失函数，表示对训练数据的拟合程度，它鼓励我们的模型尽量去拟合训练数据，使得模型会有比较小的 bias。第二项为正则化，它鼓励使用更简单的模型，因为模型简单后，有限数据拟合处理结果的随机性比较小，不容易过拟合，使得模型的预测更加稳定

其中

- $\Omega(f_t)$ 为正则项，包括 L1、L2 正则
- 对应 $f(x)$，xgboost 利用泰勒展开三项，做了一个近似

### 模型学习与训练误差

类似 GBDT，xgboost 也需要将多棵树的得分累加得到最终的预测得分
$$
\hat y_i^0 = 0\\
\hat y_i^1 = f_1(x_i) = \hat y_i^0 + f_1(x_i)\\
\hat y_i^2 = f_1(x_i) + f_2(x_i) = \hat y_i^1 + f_2(x_i)\\
\hat y_i^t = \sum_m f_m(x_i) = \hat y_i^{(m-1)} + f_m(x_i)
$$
$f_m(x)$ 如何选择呢？当然是选择能使目标函数尽量大的降低的 $f_m(x)$

即
$$
argmin_{f_m(x)}Obj^m = L(\phi) = \sum_iL(y_i - \hat y_i) + \sum_m\Omega(f_m)\\
=\sum_iL(y_i,\hat y_i^{m-1}+f_m(x_i))+\Omega(f_m) + constant
$$
损失函数不是二次函数怎么做？可以利用泰勒展开，近似为二次
$$
目标：\ Obj^m = \sum_iL(y_i,\hat y_i^{m-1}+f_m(x_i))+\Omega(f_m) + constant\\
用泰勒展开近似原来的目标\qquad\qquad\qquad\qquad\qquad\qquad\qquad\\
\qquad 泰勒展开：f(x+\triangle x)\simeq f(x) + f'(x)\triangle x + \frac12f''(x)\triangle x^2\\
\qquad\quad 定义：g_i=\partial_{\hat y^{m-1}}L(y_i,\hat y^{m-1}),h_i = \partial^2_{\hat y^{m-1}}L(y_i,\hat y^{m-1})\qquad\quad\\
\rightarrow Obj^m\simeq\sum_i[L(y_i,\hat y_i^{m-1})+g_i f_m(x_i)+\frac12h_if_m^2(x_i)] +\Omega (f_m)
$$

- 泰勒二阶展开 f 里的 x 对应于目标函数的 $\hat y_i^{m-1}$
- $\triangle x$ 对应于 $f_m(x_i)$
- f 对 x 求导数，即为目标函数对 $\hat y_i^{m-1}$ 求导

> 泰勒公式是一个用函数在某点的信息描述其附近取值的公式，就是说可由利用泰勒多项式的某些次项做原函数的近似

考虑到第 m 棵树是根据前面的 m-1 棵树的残差得到的，相当于前 m-1 棵树的值 $\hat y_i^{t-1}$ 是已知的。也就是说 $L(y_i,\hat y_i^{t-1})$ 对目标函数的优化无影响，可连同常数项一起去掉，可得
$$
Obj^m = \sum_i[g_if_m(x_i) + \frac12h_if_m^2(x_i)] + \Omega(g_m)
$$
这时，目标函数只依赖于每个数据点在损失函数上的一阶导数 g 和二阶导数 h（这就是 xgboost 和 gbdt 在目标函数上的不同，xgboost 的目标函数保留了泰勒展开的二次项）

总的原则就是把样本分配到叶子节点会对应一个目标函数，优化过程就是目标函数优化，也就是分裂节点到叶子不同的组合，不同的组合对应不同的损失函数，所以的优化围绕这个思想展开

### 树的复杂度

规则

- 用叶子节点集合以及叶子节点得分表示
- 每个样本都落在叶结点上
- $q(x)$ 表示样本 x 在某个叶结点上，$w_{q(x)}$ 是该节点的打分，即模型预测值

当我们把树分成结构部分 q 和叶子权重部分 w 后，结构函数 q 把输入映射到叶子的索引号上面去，而 w 给定了每个索引号对应的叶子分数是多少

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\20200103122730157.png" alt="img" style="zoom: 50%;" />

树的复杂度包含了两个部分

1. 树里面叶子节点个数 T
2. 树上叶子节点的得分 w 的 L2 的模的平方

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\20170228153006200" alt="img" style="zoom: 67%;" />

代入目标函数为
$$
Obj^m\simeq\sum_i[L(y_i,\hat y_i^{m-1})+g_i f_m(x_i)+\frac12h_if_m^2(x_i)] +\Omega (f_m)\\
= sum_i[g_iw_{q(x_i)} +\frac12h_iw_{q(x_i)}^2] +\gamma T + \lambda\frac12\sum_j w_j^2\\
=\sum_j[(\sum_{i\in R_j}g_i)w_j + \frac12(\sum_{i\in R_j}h_i + \lambda)w_j^2] + \gamma T
$$
$R_j = \{i|q(x_i)=j\}$ 为每个叶结点 j 上面样本下标的集合

定义
$$
G_j = \sum_{i\in R_j}g_i\quad H_j = \sum_{i\in R_j}h_i
$$
代入目标函数得
$$
Obj^t = \sum_j[G_jw_j + \frac12(H_j + \lambda)w_j^2]+\lambda T
$$
对 $w_j$ 求导等于 0，得
$$
w_j^* = -\frac {G_j}{H_j+\lambda}
$$
代入得
$$
Obj = -\frac12\sum_j\frac{G_j^2}{H_j+\lambda} + \gamma T
$$

## 打分函数计算

Obj 代表了当我们指定一个树的结构的时候，我们在目标最多减少多少，称为结构分数

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\20160421110535150" alt="img" style="zoom: 50%;" />

### 分裂节点

节点的分裂有两种方式

（1）枚举所有不同树结构的贪心法

现在的情况是只要知道树的结构，就能得到一个该结构下的最好分数，那如何确定树的结构？

想当然的一个方法是枚举，利用打分函数寻找出一个最优结构的树，接着加入模型中，不断重复，但是状态太多的话，计算量会非常大，怎么解决呢？

先试一下贪心算法，从树深度 0 开始，每一节点都遍历所有特征，比如年龄、性别等，然后对某个特征，先按照特征里的值进行排序，然后线性扫描该特征进而确定最好的分割点，最后对所有特征进行分割后，选择增益最高的那个特征，如何计算增益呢?
$$
Obj = -\frac12\sum_j\frac{G_j^2}{H_j+\lambda} + \gamma T
$$
目标函数中 $\frac{G^2}{H+\lambda}$ 表示每一个叶子节点对当前模型损失的贡献程度，融合一下，得到 Gain 的计算表达式
$$
Gain = \frac12[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \lambda
$$
中括号中前两项表示按照分割点分割后的左右子树的分数，第三项表示不分割我们可以拿到的分数，$\lambda$ 表示加入新叶子节点引入的复杂度代价

这里是将某个特征按照特征值进行排序，再设置值 a，枚举所有 $x<a,x>a$ 这样的条件，计算每个分割点分割之后的增益值，选择最大的作为最优分割点（同信息增益，只是增益函数不同）

值得注意的是引入分割不一定会是情况变好，所以有一个引入新叶子的惩罚项，优化这个目标对应了树的剪枝，当引入分割带来的增益小于阈值的时候，就忽略这个分割

也就是说，当引入某项分割，结果分割之后得到的分数减去不分割的分数得到的结果很小，但因此得到的模型复杂度过高，那不如不分割，

**算法——分割点查找的贪心算法**
$$
\begin{aligned}
&input:R,当前节点中样本点的集合\\
&input:d,特征\\
&gain = 0\\
&G = \sum_{i\in R}g_i,H=\sum_{i\in R}h_i\\

& for\ m\ in\ range(M):\\\
&\qquad G_L = 0,H_L=0\\
 &\qquad for\ j\ in\ sorted(R,by\ x_{jk}):\\
 &\qquad\qquad G_L += g_j,H_L += h_j\\
 &\qquad\qquad G_R = G - G_L,H_R += H - H_L\\
 &\qquad\qquad score = max(score,\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda})
 \end{aligned}
$$
（2）近似算法——针对数据太大，不能直接计算

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\20170228144525979" alt="img" style="zoom:80%;" />

简单的说，把样本从根分配到叶子节点，就是个排列组合。不同的组合对应的 cost 不同，求最好的组合就要尝试，以为穷举是不可能的，所以才处理贪婪法。不从头到尾，就看当下节点怎么分配最好。才有了 exact greddy 方法。

对于 $\hat y_i^t = \hat y_i^{t-1} + f_t(x_i)$，通常会写成 $\hat y_i^t = \hat y_i^{t-1} + \epsilon f_t(x_i),\epsilon$ 为缩减因子，为了避免过拟合。这意味着并不是在每一步都做充分的优化，也会为之后的训练提供机会。

在分裂的时候，每次节点分裂，影响损失函数的只有这个节点的样本，因而每次分裂，计算分裂的增益只需要关注打算分裂的那个节点的样本，分裂之后，形成一棵树，然后在这棵树预测的基础上取最优进一步分裂/建树。

停止条件：树的最大深度，节点分裂后的增益，样本权重和（min_child_weight），节点样本数

> 出处：[通俗理解kaggle比赛大杀器xgboost](https://blog.csdn.net/v_JULY_v/article/details/81410574)







