EM 算法是一种迭代算法，用于含有隐变量（hidden varaiable）的概率模型参数的极大似然估计，或极大后验概率估计。每次迭代由两步组成：E 步，求期望（expectation）；M 步，求极大（maximization）。因此又称为期望极大算法（expectation maximization）

# EM 算法的引入

概率模型又是既含有观测变量（observable variable），又含有隐变量或潜在变量（latent variable）。如果概率模型的变量都是观测变量，那么给定数据，可以直接用极大似然估计，或贝叶斯估计估计模型参数。但是当模型含有隐变量时，就不能简单地使用这些估计方法。EM 算法就是含有隐变量的概率模型参数的极大似然估计或极大后验概率估计法

## EM 算法

**例 三硬币模型** 假设有 3 枚硬币，记作 A,B,C。这些硬币正面出现的概率为 $\pi,p,q$，进行如下实验：先抛 A，根据结果选择 B 或 C，正面选 B，反面选 C，然后抛选出的硬币，出现正面记为 1，反面记为 0，独立重复 n 次实验，结果如下
$$
1,1,0,1,0,0,1,0,1,1
$$
假设只能观测到掷硬币的结果，不能观测到掷硬币的过程，问如何估计三硬币正面出现的概率，即模型的参数

解 三硬币模型可写作
$$
P(y|\theta) = \sum_zP(y,z|\theta) = \sum_zP(z|\theta)P(y|z,\theta)\\
=\pi p^y(1-p)^{1-y} + (1-\pi)p^y(1-p)^{1-y}\tag{9.1}
$$
y 表示观测变量，表示依次实验观测的结果为 1 或 0；z 表示隐变量，表示为观测到的掷硬币 A 的结果。$\theta=(\pi,p,q)$ 为模型参数，该模型为以上数据的生成模型

将观测数据表示为 $Y=(Y_1,Y_2,\cdots,Y_n)^T$，为观测数据表示为 $Z=(Z_1,Z_2,\cdots,Z_n)^T$，则观测数据的似然函数为
$$
P(Y|\theta) = \sum_ZP(Z|\theta)P(Y|Z,\theta)\tag{9.2}
$$
即
$$
P(Y|\theta) = \prod_j[\pi p^y_j(1-p)^{1-y_j} + (1-\pi)p^y_j(1-p)^{1-y_j}]\tag{9.3}
$$
考虑求模型参数 $\theta = (\pi,p,q)$ 的极大似然估计
$$
\hat \theta = argmax_\theta logP(Y|\theta)\tag{9.4}
$$
这个问题没有解析解，只有通过迭代的方式求解

EM 算法首先选取参数的初值，记作 $\theta^0 = (\pi^0,p^0,q^0)$，然后通过下面的步骤迭代计算参数的估计值，直到收敛为止。第 i 次迭代参数的估计值为 $\theta^i = (\pi^i,p^i,q^i)$。第 i + 1 次迭代如下

E 步：计算在模型参数 $\theta^i$ 下观测数据 $y_j$ 来自掷硬币 B 的概率
$$
\mu^{i + 1} = \frac{\pi^i(p^i)^{y_j}(1-p^i)^{1-y_j}}{\pi^i(p^i)^{y_j}(1-p^i)^{1-y_j} + （1-\pi^i）(q^i)^{y_j}(1-q^i)^{1-y_j}}\tag{9.5}
$$
M 步：计算参数的新估计值
$$
\pi^{i+1} = \frac1n\sum_j\mu_j^{i+1}\tag{9.6}
$$

$$
p^{i+1}=\frac{\sum_j\mu_j^{i+1}y_j}{\sum_j\mu_j^{i+1}}\tag{9.7}
$$

$$
q^{i+1}=\frac{\sum_j(1 - \mu_j^{i+1})y_j}{\sum_j(1-\mu_j^{i+1})}\tag{9.8}
$$

假设模型参数的初值为 $\pi^0 = 0.5,p^0 = 0.5,q^0=0.5$

由 9.5，对 $y_j=1,y_j=0$ 均有 $\mu_j^1=0.5$

利用公式 9.6 ~ 9.8，得到
$$
\pi^1 = 0.5,p^1 = 0.6,q^1=0.6
$$
继续迭代可得，参数 $\theta$ 的最大似然估计
$$
\hat\pi = 0.5,\hat p=0.6,\hat 1 = 0.6
$$
如果取初值 $\pi^0 = 0.4,p^0 = 0.6,q^0=0.7$，得到的结果为 $\hat\pi = 0.406,\hat p=0.537,\hat 1 = 0.643$，也就是说，EM 算法与初值的选择有关，不同的初值可能得到不同的参数估计值

一般地，Y 和 Z 连在一起称为完全数据（complete-data），给定 Y，概率分布为 $P(Y|\theta)$，对数似然函数为 $L(\theta)=logP(Y|\theta)$；假设 Y，Z 的联合概率分布为 $P(Y,Z|\theta)$，对数似然函数为 $logP(Y,Z|\theta)$

EM 算法通过迭代求 $L(\theta)=log P(Y|\theta)$ 的极大似然估计

**算法 9.1**

（1）选择参数的初值 $\theta^0$，开始迭代——初值可任意选择，但算法对初值敏感

（2）E 步：记 $\theta^i$ 为第 i 次迭代参数 $\theta$ 的估计值，在第 i+1 次迭代的 E 步，计算
$$
Q(\theta,\theta^i) = E_Z[logP(Y,Z|\theta)|Y,\theta^i]\\
=\sum_ZlogP(Y,Z|\theta)P(Z|Y,\theta^i)\tag{9.9}
$$
$P(Z|Y,\theta^i)$ 是在给定观测数据 Y 和当前的参数估计 $\theta^i$ 下 Z 的条件概率分布，$\theta$ 表示要极大化的参数，$\theta^i$ 表示参数的当前估计值

（3）M 步：求使 $Q(\theta,\theta^i)$ 极大化的 $\theta$，确定第 i+1 次迭代的参数的估计值 $\theta^{i+1}$
$$
\theta^{i+1} = argmax_\theta Q(\theta,\theta^i)\tag{9.10}
$$
（4）重复第 2 和第 3 步，直到收敛——收敛条件，一般是对较小的正数 $\epsilon_1,\epsilon_2$，满足 $||\theta^{i+1} - \theta^{i}|| < \epsilon_1 或 ||Q(\theta^{i+1},\theta^i) - Q(\theta^i,\theta^i)|| < \epsilon_2$ 则停止迭代

函数 $Q(\theta,\theta^i)$ 是 EM 算法的核心，称为 Q 函数

**定义 9.1 Q 函数** 完全数据的对数似然函数 $log P(Y,Z|\theta)$ 关于在给定观测数据 Y 和当前参数 $\theta^i$ 下对未观测数据 Z 的条件概率分布 $P(Z|Y,\theta^i)$ 的期望称为 Q 函数

## EM 算法的导出

我们面对一个含有隐变量的概率模型，目标是极大化观察数据Y关于参数θ的对数似然函数即极大化
$$
L(\theta) = logP(Y|\theta) = log\sum_zP(Y,Z|\theta)\\
=log[\sum_zP(Y|Z,\theta)P(Z|\theta)]
$$
事实上，EM算法是通过迭代逐步近似极大化L(θ)的，假设在第i次迭代后θ的估计值$θ^i$，我们希望新估计值θ能使L(θ)增加，即$L(θ) > L(θ^i)$，并逐步达到极大值，为此，考虑两者的差：
$$
L(\theta) - L(\theta^i) = log[\sum_zP(Y|Z,\theta)P(Z|\theta)] - logP(Y|\theta^i)
$$
利用Jensen不等式得到其下界：
$$
L(\theta) - L(\theta^i) = log[\sum_zP(Y|Z,\theta)P(Z|\theta)] - logP(Y|\theta^i)\\
=log[\sum_zP(Y|Z,\theta)\frac{P(Z|\theta) P(Z|Y,\theta^i)}{ P(Z|Y,\theta^i)}]- logP(Y|\theta^i)\\=
log[\sum_z P(Z|Y,\theta^i)\frac{P(Y|Z,\theta)P(Z|\theta)}{ P(Z|Y,\theta^i)}]- logP(Y|\theta^i)\\≥
\sum_z P(Z|Y,\theta^i)log[\frac{P(Y|Z,\theta)P(Z|\theta)}{ P(Z|Y,\theta^i)}]- logP(Y|\theta^i)\\=
\sum_z P(Z|Y,\theta^i)log[\frac{P(Y|Z,\theta)P(Z|\theta)}{ P(Z|Y,\theta^i)P(Y|\theta^i)}]
$$
令
$$
B(\theta,\theta^i) = L(\theta^i) +
\sum_z P(Z|Y,\theta^i)log[\frac{P(Y|Z,\theta)P(Z|\theta)}{ P(Z|Y,\theta^i)P(Y|\theta^i)}]
$$
则
$$
L(\theta) ≥B(\theta,\theta^i) 
$$
所以，任何可以使 $B(θ，θ^i)$ 增大的θ，也可以使 L(θ) 增大，为了使 L(θ) 有尽可能大的增长，选择$θ_{i + 1}$使$B(θ，θ^i)$达到极大，即
$$
\theta^{i + 1} = argmax_\theta B(\theta,\theta^i)
$$
现在求θ的表达式，省去对θ的极大化而言是常数的项，则有

$$
\theta^{i+1} = argmax_\theta[L(\theta^i) +
\sum_z P(Z|Y,\theta^i)log[\frac{P(Y|Z,\theta)P(Z|\theta)}{ P(Z|Y,\theta^i)P(Y|\theta^i)}]]\\=
argmax_\theta[\sum_z P(Z|Y,\theta^i)logP(Y|Z,\theta)P(Z|\theta)]\\=
argmax_\theta[\sum_z P(Z|Y,\theta^i)logP(Y,Z|\theta)]\\=
argmax_\theta Q(\theta,\theta^i)
$$
以上即为 EM 算法的一次迭代，即求 Q 函数及其极大化，EM 算法是通过不断求解下界的极大化逼近求解对数似然函数极大化的算法。

下面给出EM算法的直观解释，图中上方曲线为 L(θ)，下方曲线为$B(θ，θ^i)$，$B(θ，θ^i)$为对数似然函数 L(θ) 的下界，两个函数在点 $θ=θ^i$ 处相等，EM 算法找到下一个点 $θ^{i + 1}$ 使函数 $B(θ，θ^i)$ 极大化，也是 Q 极大化，这时 $L≥B$，函数 B 的增加，保证对数似然函数  L在每次迭代中也是增加的，EM算法在点 $θ^{i + 1}$ 重复计算 Q 函数值，进行下一次迭代，在这个过程中，L 不断增大，从图可以推断出 EM 算法不能保证找到全局最优值

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210429164417495.png" alt="image-20210429164417495" style="zoom:50%;" />

## 在非监督学习中的应用

EM 算法可以用于生成模型的非监督学习，生成模型由联合概率分布 P(X，Y) 表示，可以认为非监督学习训练数据是联合概率分布产生的数据

## 例

假设这样一个抛硬币的测试, 我们有两个硬币A和B, 它们正面朝上的概率分别为 $\theta_A,\theta_B$, 我们重复以下操作五次:

- 从AB中随机抽一个硬币
- 抛该硬币10次并记录结果

接着我们需要从这个实验中得到对 $\theta_A,\theta_B$ 的估计.

极大似然估计

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\v2-ee4d9c71b608bf571f7fb98b5e34a7bc_720w.jpg" alt="img" style="zoom:50%;" />

现在来给这个游戏加点难度. 现在假设我们不知道每次操作的时候我们抛的到底是硬币A还是B, 也就是说, 变量 z 变得不可观测了(hidden variable). 此时我们常称 z 为**隐变量(hidden variable / latent factor)**. 这种情况在参数估计中叫做不完全数据. 此时我们就无法直接用极大似然估计来估算概率了, 为此就有了EM算法.

使用em算法流程求解

如上所述, 我们的问题变成了不完全数据下的参数估计, 那么一个简单的想法就是**将这种不完全数据给补全了**, 那我们的问题不就可以归结到完全数据的参数估计了嘛.

于是, 一种很简单的迭代式的想法就是:

1. 先初始化参数 $\hat\theta^t=(\hat\theta_A^t,\hat\theta_B^t)$
2. 对测试中每次操作, 基于现有的参数猜想它是由A还是B抛出的.
3. 根据上一步的猜想, 我们将不完全的数据补全了, 此时就可以用极大似然估计来更新参数 $\theta^{t+1}$
4. 重复前两步步骤直到收敛.

其实到这里, EM算法的大概思路就已经出来了. 第三步中的问题其实就是上一节所介绍的, 那么重点就在于第二步中我们如何猜想. 实际上EM算法基于现有的参数 $\hat\theta^t$ 来对缺失数据(隐变量)的每种可能都计算出概率值, 然后用这种概率对所有可能的补全数据加权. 最后我们的极大似然就基于这种调整了权重的训练样本.

下面我们举个例子说明一下:

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\v2-2a0cc644751d6e4f778fa589756cf07c_720w.jpg" alt="img" style="zoom:50%;" />

在上图中, 我们按步骤标号来解释:

1. 初始化参数 $\hat\theta_A^0=0.60,\hat\theta_B^0=0.50$

2. E-step, 我们对缺失的数据进行补全. 例如对于第一次操作，$z_1=A$ 的概率为0.45, 而 $z_1=B$ 的概率为0.55. 其他四次实验样本也是类似的操作, 然后我们就得到右边这样一个调整了权重的训练样本.
   $$
   以第一组样本为例：
   H:5,T:5\\
   P_A = \theta_A^5(1-\theta_A)^5 = 0.45\\
   P_B = \theta_B^5(1-\theta_B)^5 = 0.55\\
   coin_A = ((0.45*5)^H,(0.45*5)^T)\\
   coin_B = ((0.55*5)^H,(0.55*5)^T)
   $$

3. M-step, 基于上面的数据用极大似然估计来更新参数.

4. 重复上述两步若干次到收敛后, 输出结果.

在上面的步骤中E-step得名于Expectation(期望), 其实也就是我们通常没必要显示计算出数据补全的概率分布, 而只需要计算期望的显著统计量. 而M-step得名于Maximization, 也就是这一步会去最大化期望的对数似然.

我们先回到最初的设定, 即用极大似然解决的完全数据问题, 此时我们的目标函数 $logP(x,z|\theta)$ 有一个全局最优点, 也就是说该问题通常有一个解析解.

与之相反的是, 在不完全数据的情况下, 目标函数 $logP(x|\theta)$ 有多个局部最优而没有闭式解.

为了解决这个问题, EM算法的做法**将这个困难的优化问题化解为一系列更简单的子问题**, 具体而言, 这些子问题都有一个全局最优且有闭式解, 而这些依次解决的子问题的解 $\hat\theta^1,\hat\theta^2$ 保证会收敛于原问题的一个局部最优.

在E-step中, 算法选择了一个函数 $g_t$ 作为目标函数 $logP(x|\theta)$ 的下界, 也就是 $g_t(\hat\theta^t) = logP(x|\hat\theta^t)$，而在M-step中, 算法就针对最大化 $g_t$ 函数选择了一个新的参数 $\theta^{t+1}$， 因为我们的下界函数 $g_t$ 对应了参数为 $\hat\theta^t$ 时的目标函数, 我们也就有
$$
logP(x|\hat\theta^t) = g_t(\hat\theta^t)\leq g_t(\hat\theta^{t+1})=logP(x|\theta^{t+1})
$$
那么目标函数也就能在迭代中不断下降直到收敛到局部最优，为了避免局部最优，通常会以不同的初始点多次进行 EM 算法来打破模型的对称性，并选择其中最优的结果

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210429171341059.png" alt="image-20210429171341059" style="zoom:50%;" />

蓝线为真实的目标函数，绿线为不断逼近局部最优的下界函数

参考：猴子也能理解的EM算法-Lunarnai-知乎-https://zhuanlan.zhihu.com/p/60376311



