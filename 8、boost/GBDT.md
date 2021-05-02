

# 提升方法

# Boosting Decision Tree(提升树)

提升树算法：拟合残差

1. 初始化$f_0(x) = 0$

2. 对于决策树m=1,2,...,M

   - 计算残差

     $r_{mi} = y_i - f_{m-1}(x),i=1,2,...,N$

   - 拟合残差$r_{mi}$学习一个回归树，得到$h_m(x)$

   - 更新$f_m(x) = f_{m-1} + h_m(x)$

3. 得到回归问题的提升树

   $f_M(x) = \sum_{m=1}^M h_m(x)$

   m为决策树编号，M为决策树数量

   假设前一轮得到的强学习器是$f_{t-1}(x)$

   损失函数是$L(y,f_{t-1}(x))$

   本轮迭代的目标是找到一个弱学习器$h_t(x)$

   最小化本轮的损失$L(y,f_t(x))=L(y,f_{t-1}(x)+h_t(x))$

   当采用平方损失函数时
   $$
   L(y,f_t(x))=L(y,f_{t-1}(x)+h_t(x))\\
   =(y - f_{t-1}(x) - h_t(x))^2\\
   =(r-h_t(x))^2
   $$
   $r = y - f_{t-1}(x)$ 即当前模型拟合数据的残差，所以提升树只需要当初拟合模型的残差

# GBDT

## 简介

GBDT 无论用于分类还是回归，使用的都是 CART 回归树。这是因为 GBDT 每轮的训练是在上一轮训练模型的负梯度值基础上进行的。这就要求每轮迭代的时候，真实标签减去弱分类器的输出结果是有意义的，即残差有意义，而如果用分类数，类别相减没有意义。对于这样的问题，可以采用两种方法解决：

- 采用指数损失函数，这样就和 AdaBoost 一样了，可以解决分类问题
- 使用类似逻辑回归的对数似然损失函数，这样就可以通过结果的概率值与真实概率值的差距当作残差来拟合

## 二分类算法

### 逻辑回归对数损失函数

预测函数
$$
h_w(x) = \frac1{1 + e^{-w^tx}}
$$
它表示结果取 1 的概率，因此对于输入 x 分类结果为
$$
P(Y=1|x;w) = h_w(x)\\
P(Y=0|x;w) = 1- h_w(x)
$$
上式综合起来可以写成
$$
P(Y=y|x;w) = (h_w(x))^y(1-h_w(x))^{(1-y)}
$$
似然函数为
$$
L(w) = \prod_i(h_w(x_i))^{y_i}(1-h_w(x_i))^{(1-y_i)}
$$
取对数为
$$
J(w) = \frac1NL(w) = \frac1N\sum_i[y_ilog(h_w(x_i))+(1-y_i)log(1-h_w(x_i))]
$$
最大似然估计就是求使得 L 取最大值时的 w，最为求最小值，可以使用梯度下降法求解，得到的 w 就是要求的最佳参数
$$
J(w) = -\frac1NL(w) = -\frac1N\sum_i[y_ilog(h_w(x_i))+(1-y_i)log(1-h_w(x_i))]
$$

### 二分类原理

逻辑回归单样本的损失函数为
$$
L(w) = -y_ilog\hat y_i -(1-y_i)log(1-\hat y_i)
$$
$\hat y_i=h_w(x_i)$ 表示逻辑回归的预测结果。假设第 M 步迭代之后当前学习器为 $F(x)=\sum_m h_m(x)$，将 $\hat y_i$ 替换为 F(x) 代入上式，为
$$
L(y_i,F(x_i)) = y_ilog(1+e^{-F(x_i)})+(1-y_i)[F(x_i)+log(1+e^{-F(x_i)})]
$$
第 m 棵树对应的响应（负梯度即伪残差）为
$$
r_{m,i} = -|\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}|_{F(x)=F_{m-1}(x)} = y_i - \frac1{1+e^{-F(x_i)}}=y_i-\hat y_i
$$
对于生成的决策树，计算各个叶子节点的最佳残差拟合值为：
$$
c_{m,j} = argmin_c\sum_{x_i\in R_{m,j}}L(y_i,F_{m-1(x_i)}+c)
$$
上式没有闭式解（closed form solution），一般用近似值代替
$$
c_{m,j} = \frac{\sum_{x_i\in R_{m,j}}r_{m,i}}{\sum_{x_i\in R_{m,i}}(y_i - r_{m,i})(1-y_i+r_{m,i})}
$$
推导

假设仅有一个样本
$$
L(y_i,F(x_i)) = -y_iln(\frac1{1+e^{-F(x_i)}})-(1-y_i)ln(1-\frac1{1+e^{-F(x_i)}})
$$
令 $P_i = \frac1{1+e^{-F(x_i)}}$，求一阶导
$$
\frac{\partial L}{\partial F(x)} = \frac{\partial L}{\partial P_i}\frac{\partial P_i}{\partial F(x)}\\
=-(\frac{y_i}{P_i}-\frac{1-y_i}{1-P_i})P_i(1-P_i)\\
=P_i-y_i
$$
求二阶导
$$
\frac{\partial^2 L}{\partial^2F(x)} = (P_i-y_i)'\\
=P_i(1-P_i)
$$
对于 $L(y_i,F(x)+c)$ 的泰勒二阶展开式为
$$
L(y_i,F(x)+c) = L(y_i,F(x))+\frac{\partial L}{\partial F(x)}\cdot c + \frac12\frac{\partial^2 L}{\partial^2F(x)}\cdot c^2
$$
L 取极值时，上述二阶表达式中的 c 为：
$$
c = -\frac b{2a} = -\frac{\frac{\partial L}{\partial F(x)}}{2(\frac12\frac{\partial^2 L}{\partial^2F(x)})}\\
=\frac{y_i-P_i}{P_i(1-P_i)}\\
=\frac{r_i}{(y_i-r_i)(1-y_i+r_i)}
$$
**二分类算法**

（1）初始化弱学习器 $F_0(x)$
$$
F_0(x)=log\frac{P(Y=1|x)}{1-P(Y=1|x)}
$$
$P(Y=1|x)$ 是训练样本中 y=1 的比例，利用先验信息来初始化学习器

（2）建立 M 棵分类回归树

​	a. 对 $i=1,2,\cdots,N$，计算第 m 棵树对应的响应值
$$
r_{m,i} = -|\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}|_{F(x)=F_{m-1}(x)} = y_i - \frac1{1+e^{-F(x_i)}}
$$
​	b. 对 $i=1,2,\dots,N$ ，利用 CART 回归树拟合数据 $(x_i,r_{m,i})$，得到第 m 棵树，对应的叶结点区域为 $R_{m,j},j=1,2,\cdots,J_m,J_m$ 为第 m 棵树叶结点的个数

​	c. 对应 $J_m$ 个叶结点区域，计算最佳拟合值 $c_{m,j}$

​	d. 更新强学习器 $F_m(x)$
$$
F_m(x) = F_{m-1}(x) + lr*\sum_jc_{m,j}I(x\in R_{m,j})
$$
（3）得到最终的强学习器 $F_M(x)$ 的表达式
$$
F_M(x) = F_0(x) + \sum_m\sum_j c_{m,j}I(x\in R_{m,j})
$$
对应逻辑回归而言 $log\frac p{1-p}=w^Tx,p=P(Y=1|x)$，逻辑回归用一个线性模型拟合 $Y=1|x$ 这个时间的对数几率（odds）$log\frac p{1-p}$。二元 GBDT 分类算法和逻辑回归一样，用一系列梯度提升树取拟合这个对数几率，分类模型可表达为
$$
P(Y=1|x) = \frac1{1+e^{-F_M(x)}}
$$

### sklearn 实现

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

'''
调参：
loss：损失函数。有deviance和exponential两种。deviance是采用对数似然，exponential是指数损失，后者相当于AdaBoost。
n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管

由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
max_depth：CART最大深度，默认为None
min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
min_leaf_nodes：最大叶子节点数
'''
```

> 出处：[深入解析GBDT二分类算法（附代码实现）](https://mp.weixin.qq.com/s/XLxJ1m7tJs5mGq3WgQYvGw)

## GBDT 回归树

算法流程与分类树相同，不同的是损失函数，回归树的损失函数为
$$
L(y_i,F(x_i)) = \frac12(y_i-F(x_i))^2
$$
负梯度的值为
$$
r_{m,i} = -|\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}|_{F(x)=F_{m-1}(x)}
$$
初始化模型
$$
F_0(x) = \bar y\\
$$


