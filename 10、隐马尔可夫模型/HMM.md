隐马尔可夫模型（HMM）是可用于标注问题的统计学习模型，描述由隐藏的马尔科夫链随机生成观测序列的过程，属于生成模型

# 基本概念

## 定义

隐马尔可夫模型是关于时序的概率模型，描述由一个隐藏的马尔科夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。隐藏的马尔可夫链随机生成的状态的序列，称为状态序列（state sequence）；每个状态生成一个观测，而由此产生的观测的随机序列，称为观测序列（observation sequence）。序列的每个位置可以看作一个时刻

隐马尔可夫模型由初始概率分布、状态转移概率分布以及观测概率分布确定。模型定义如下：

设 Q 是所有可能的状态的集合， V 是所有可能的观测集合
$$
Q = \{q_1,q_2,\cdots,q_N\},V = \{v_1,v_2,\cdots,v_M\}
$$
$I$ 是长度为 T 的状态序列，O 是对应的观测序列
$$
S = \{s_1,s_2,\cdots,s_T\},O = \{o_1,o_2,\cdots,o_T\}
$$
下图为隐马尔可夫链过程(下图 i 其实为上面的 s 表示状态)

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210512101531574.png" alt="image-20210512101531574" style="zoom:50%;" />

​																		图1

A 是状态转移概率矩阵：
$$
A = [a_{ij}]_{N*N}\\
a_{ij} = P(i_{t+1} = q_j|i_t=q_i),i\in[1,N];j\in[1,N]
$$
是在时刻 t 处于状态 $q_i$ 的条件下在时刻 t + 1 转移到状态 $q_j$ 的概率

B 是观测概率矩阵：
$$
B  = [b_j(k)]_{N*M}\\
b_j(k) = P(o_t = v_k|s_t = q_j),k\in[1,M];j\in[1,N]
$$
是在时刻 t 处于状态 $q_j$ 的条件下生成观测 $v_k$ 的概率

$\pi$ 是初始状态概率向量：
$$
\pi = (\pi_i)\\
\pi_i = P(s_1=q_i),i\in[1,N]
$$
是时刻 t = 1 处于状态 $q_i$ 的概率

顺着图 1 隐马尔可夫链走，首先 t=1 时刻初始状态没有前驱状态，发生概率有 $\pi$ 决定
$$
P(s_1=q_i) = \pi_i
$$
接着对 $t\geq 2$，状态 $s_t$ 由前驱状态 $s_{t-1}$ 转移而来，转移概率可由矩阵 A 得到
$$
P(s_t = q_j|s_{t-1}=q_i) = A_{ij}
$$
状态序列的概率为上述两个式子的乘积
$$
P(S)=P(s_1,\cdots,s_T) = P(s_1)\prod_{t=2}P(s_t|s_{t-1})
$$
最后，对每个 $s_t=q_i$，都会计算 $o_t=v_j$，概率由矩阵 B 得到
$$
P(o_t=v_j|s_t=q_i)=B_{q_i}(v_j)
$$
那么给定长 T 的状态序列 $S$，对应 O 的概率就是上式的累积形式
$$
P(O|S) = \prod_tP(o_t|s_t)
$$
显隐状态的联合概率
$$
\begin{align}P(O,S) &= P(S)\cdot P(O|S)\\
&=P(s_1)\prod_{i=2}P(s_t|s_{t-1})\prod_tP(o_t|s_t)
\end{align}
$$
将其中每个 $s_t,o_t$ 对应实际发生的序列的 $q_i,v_j$，就能代入 $(\pi,A,B)$ 中的相应元素，从而计算出任意序列的概率了

隐马尔可夫模型由 $\pi$、A 和 B 决定，$\pi$ 和 A 确定了隐藏的马尔可夫链，生成不可观测的状态序列，B 确定了如何从状态生成观测，与状态序列综合确定了如何产生观测序列。因此一般用三元符号表示隐马尔可夫模型 $\lambda$
$$
\lambda = (A,B,\pi)
$$
隐马尔可夫模型做了两个基本假设：

1. 齐次马尔可夫性假设，即假设隐藏的马尔可夫链在任意时刻 t 的状态只依赖于前一时刻的状态，与其他时刻的状态及观测无关，与时刻 t 也无关
   $$
   P(s_t|s_{t-1},o_{t-1},\cdots,s_1,o_1) = P(s_t|s_{t-1})
   $$

2. 观测独立性假设，即假设在任意时刻的观测只依赖于该时刻的马尔可夫链的状态，与其他观测即状态无关
   $$
   P(o_t|s_T,o_T,\cdots,s_{t+1},o_{t+1},s_t,s_{t-1},o_{t-1},\cdots,s_1,o_1) = P(o_t|s_t)
   $$

隐马尔可夫模型可以用于标注，这是状态对应着标记。标注问题是给定观测的序列预测其对应的标记序列。可以加速标注问题的数据是由隐马尔可夫模型生成的。这样就可以利用 HMM 的学习和预测算法进行标注

例 1

假设有 4 个盒子，每个盒子里都装有红白两种颜色的球，盒子里的红白球数有下表给出

|  盒子  |  1   |  2   |  3   |  4   |
| :----: | :--: | :--: | :--: | :--: |
| 红球数 |  5   |  3   |  6   |  8   |
| 白球数 |  5   |  7   |  4   |  2   |

按照以下方法抽取，产生一个球的颜色的观测序列：开始，从 4 个盒子里以等概率随机选取 1 个盒子，从这个盒子里随机抽出 1 个球，记录其颜色后，放回；然后从当前盒子转移到下一个盒子，规则是：如果当前盒子为 1，下一个一定是 2，当前为 2 或 3，那么分别以概率 0.4 和 0.6 转移到左边或右边的盒子，当前为 4，以 0.5 的概率停留在盒子 4 或转移到盒子 3；确定转移的盒子后，在从盒子里随机抽取 1 个球，记录其颜色，放回，如此重复 5 次，得到一个观测序列：
$$
O = \{红，红，白，白，红\}
$$
这个过程中，观察者只能观测到球的颜色的序列，观测不到球是从哪个盒子取出的，即盒子的序列

这个例子中有两个随机序列，一个是盒子的序列（状态），一个是球的颜色的序列（观测）。根据所给条件，可以明确状态集合、观测集合、序列长度以及模型的三要素

盒子对应状态，状态的集合为
$$
Q = \{盒子1，盒子2，盒子3，盒子4\},N=4
$$
球的颜色对应观测，观测的集合为
$$
V = \{红，白\},M=2
$$
状态序列和观测序列长度为 5，

初始概率分布为
$$
\pi = (0.25,0.25,0.25,0.25)^T
$$
状态转移概率分布为
$$
A=\left[\begin{matrix}
0&1&0&0\\
0.4&0&0.6&0\\
0&0.4&0&0.6\\
0&0&0.5&0.5
\end{matrix}\right]
$$
观测概率分布
$$
B=\left[
\begin{matrix}
0.5&0.5\\
0.3&0.7\\
0.6&0.4\\
0.8&0.2
\end{matrix}
\right]
$$

## 观测序列的生成过程

**算法 10.1**

输入：隐马尔可夫模型 $\lambda=(A,B,\pi)$，观测序列长度 T

输出：观测序列 $O=(o_1,o_2,\cdots,o_T)$

1. 按照初始状态分布 $\pi$ 产生状态 $s_1$
2. 令 t = 1
3. 按照状态 $s_t$ 的观测概率分布 $b_{s_t}(k)$ 生成 $o_t$
4. 按照状态 $s_t$ 的状态转移概率分布 {$a_{s_ts_{t+1}}$} 产生状态 $s_{t+1}$
5. 令 t = t + 1，如果 t < T，转到 3.；否则终止

## 三个基本问题

1. 概率计算问题。给定模型 $\lambda$ 和观测序列 O，计算在模型 $\lambda$ 下观测序列 O 出现的概率 $P(O|\lambda)$
2. 学习问题。一致观测序列 O，估计模型 $\lambda$ 参数，使得在该模型下观测序列 概率 $P(O|\lambda)$ 最大，即用极大似然估计法估计参数
3. 预测问题。也称解码问题。已知模型 $\lambda$ 和观测序列 O，求对给定观测序列条件概率 $P(S|O)$ 最大的状态序列 $S$，即给定观测序列，求最有可能对应的状态序列

# 概率计算算法

本节介绍计算观测序列概率 $P(O|\lambda)$ 的前向与后向算法，先介绍概念上可行但计算上不可行的直接计算法

## 直接计算法

给定模型 $\lambda$ 和观测序列 O，计算观测序列 O 出现的概率 $P(O|\lambda)$，最直接的方法是按概率公式直接计算。通过列举所有可能的长度为 T 的状态序列 $S$，求各个状态序列 $S$ 与观测序列 O 的联合概率 $P(O,S|\lambda)$，然后对所有可能的状态序列求和，得到 $P(O|\lambda)$

状态序列 $I$ 的概率为
$$
P(S|\lambda) = \pi_{s_1}a_{s_1s_2}a_{s_2s_3}\cdots a_{s_{T-1}s_T}
$$
对固定的状态序列 $S$，观测序列 O 的概率是 $P(O|S,\lambda)$
$$
P(O|S,\lambda) = b_{s_1}(o_1)b_{s_2}(o_2)\cdots b_{s_T}(o_T)
$$
$O$ 和 $I$ 同时出现的联合概率为
$$
\begin{align}P(O,S|\lambda) &= P(O|S,\lambda)P(S|\lambda)\\
&= \pi_{s_1}b_{s_1}(o_1)a_{s_1s_2}b_{s_2}(o_2)\cdots a_{s_{T-1}s_T}b_{s_T}(o_T)
\end{align}
$$
然后，

对所有可能的状态序列 $S$ 求和，得到观测序列 O 的概率 $P(O|\lambda)$
$$
\begin{align}P(O|\lambda) &= \sum_IP(O|S,\lambda)P(S|\lambda)\\
&=\sum_{s_1,s_2,\cdots,s_T}\pi_{s_1}b_{s_1}(o_1)a_{s_1s_2}b_{s_2}(o_2)\cdots a_{s_{T-1}s_T}b_{s_T}(o_T)
\end{align}
$$
利用上述公式计算量很大，是 $O(TN^T)$ 阶的。

## 前向算法

**定义 10.2 前向概率** 给定隐马尔可夫模型 $\lambda$，定义到时刻 t 观测序列为 $o_1,o_2,\cdots,o_t$ 且状态为 $q_i$ 的概率为前向概率
$$
\alpha_t(i) = P(o_1,o_2,\cdots,o_t,s_t=q_i|\lambda)
$$
可以递推地求得前向概率 $\alpha_t(i)$ 及观测序列概率 $P(O|\lambda)$

**算法 10.2 观测序列概率的前向算法**

输入：隐马尔可夫模型 $\lambda$，观测序列 O

输出：观测序列概率 $P(O|\lambda)$

1. 初值
   $$
   \alpha_1(i) = \pi_ib_i(o_1),i\in[1,N]
   $$

2. 递推，对 $t\in[1,T-1]$
   $$
   \alpha_{t+1}(j) = [\sum_i\alpha_t(i)a_{ij}]b_j(o_{t+1}),i\in[1,N]
   $$

3. 终止
   $$
   P(O|\lambda)=\sum_i\alpha_T(i)
   $$

步骤 1 初始化前向概率，是初始时刻的状态 $i_1=q_i$ 和观测 $o_1$ 的联合概率

步骤 2 是前向概率的递推公式，计算到时刻 t + 1 观测序列为 $o_1,o_2,\cdots,o_{t+1}$ 且在时刻 t + 1 处于状态 $q_j$ 的前向概率，如下图所示，在递推公式中的方括号里，$\alpha_t(i)$ 是到时刻 t 观测到 $o_1,o_2,\cdots,o_t$ 并在时刻 t 处于状态 $q_i$ 的前向概率，乘积 $\alpha_t(i)a_{ij}$ 就是在时刻 t + 1 到达状态 $q_j$ 的联合概率。对这个乘积在时刻 t 的所有可能的 N 个状态 $q_i$ 求和，结果就是在时刻 t + 1 处于状态 $q_j$ 的联合概率。方括号里的值与观测概率 $b_j(o_{t+1})$ 的乘积恰好是到时刻 t + 1 观测到 $o_1,o_2,\cdots,o_{t+1}$ 并在时刻 t + 1 处于状态 $q_j$ 的前向概率 $\alpha_{t+1}(j)$

步骤 3 因为
$$
\alpha_T(i) = P(o_1,o_2,\cdots,o_T,s_T=q_i|\lambda)\\
\rightarrow P(O|\lambda)=\sum_i\alpha_T(i)
$$
<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210511201619903.png" alt="image-20210511201619903" style="zoom: 67%;" />

如下图所示，前向算法实际是基于 “状态序列的路径结果” 递推计算 $P(O|\lambda)$ 的算法。前向算法高效的关键是其局部计算前向概率，然后利用路径结构将前向概率 “递推” 到全局，得到 $P(O|\lambda)$。具体地，在 t = 1，计算 $\alpha_1(i)$ 的 N 个值（$i\in[1,N]$），而且每个 $\alpha_{t+1}(j)$ 的计算利用前一个时刻 N 个 $\alpha_t(i)$，减少计算量的原因在于每一次计算直接引用前一时刻的计算结果，避免重复计算。这样，利用前向概率计算 $P(O|\lambda)$ 的计算量是 $O(N^2T)$ 阶，而不是直接计算的 $O(TN^T)$

<img src="C:\Users\you\AppData\Roaming\Typora\typora-user-images\image-20210511202517868.png" alt="image-20210511202517868" style="zoom:67%;" />

例 2 考虑盒子和球模型 $\lambda$，状态集合 Q={1,2,3}，观测集合 V={红，白}
$$
A=\left[\begin{matrix}
0.5&0.2&0.3\\
0.3&0.5&0.2\\
0.2&0.3&0.5
\end{matrix}\right]
B=\left[\begin{matrix}
0.5&0.5\\
0.4&0.6\\
0.7&0.3
\end{matrix}\right]
\pi = (0.2,0.4,0.4)^T
$$
现有 T=3，O=（红，白，红），利用前向算法计算 $P(O|\lambda)$

解：

1. 计算初值
   $$
   \alpha_1(1) = \pi_1b_1(o_1)=0.1\\
   \alpha_1(2) = \pi_2b_2(o_1)=0.16\\
   \alpha_1(3) = \pi_3b_3(o_1)=0.28\\
   $$

2. 递推计算
   $$
   \alpha_2(1) = [\sum_{i\in[1,3]}\alpha_1(i)a_{i1}]b_1(o_2) = 0.154 * 0.5=0.077\\
   \alpha_2(2) = [\sum_{i\in[1,3]}\alpha_1(i)a_{i2}]b_2(o_2) = 0.184 * 0.6=0.1104\\
   \alpha_2(3) = [\sum_{i\in[1,3]}\alpha_1(i)a_{i3}]b_3(o_2) = 0.202 * 0.3=0.0606\\
   \alpha_3(1) = [\sum_{i\in[1,3]}\alpha_2(i)a_{i1}]b_1(o_3) = 0.04187\\
   \alpha_3(2) = [\sum_{i\in[1,3]}\alpha_2(i)a_{i2}]b_2(o_3) = 0.03551\\
   \alpha_3(3) = [\sum_{i\in[1,3]}\alpha_2(i)a_{i3}]b_3(o_3) = 0.05284\\
   $$

3. 终止
   $$
   P(O|\lambda)=\sum_i \alpha_3(i) = 0.13022
   $$

## 后向算法

**定义 10.3 后向概率** 给定隐马尔可夫模型 $\lambda$，定义在时刻 t 状态为 $q_i$ 的条件下，从 t + 1 到 T 的观测序列为 $o_{t+1},\cdots,o_T$ 的概率为后向概率，记作
$$
\beta_t(i) = P(o_{t+1},\cdots,o_T|s_t= q_i,\lambda)
$$
可以用递推的方法求得后向概率 $\beta_t(i)$ 及观测序列概率 $P(O|\lambda)$

**算法 10.3 观测序列概率的后向算法**

1. 初始化后向概率
   $$
   \beta_T(i) = 1,i\in[1,N]
   $$

2. 对 $t=T-1,T-2,\cdots,1$
   $$
   \beta_t(i) = \sum_ja_{ij}b_j(o_{t+1})\beta_{t+1}(j),i\in[1,N]
   $$

3. $$
   P(O|\lambda) = \sum_i\pi_ib_i(o_1)\beta_1(i)
   $$

步骤 1 初始化后向概率，对最终时刻的所有状态 $q_i$ 规定 $\beta_T(i)=1$

步骤 2 是后向概率的递推公式，如下图，为了计算在时刻 t 状态为 $q_i$ 条件下实际 t + 1 之后的观测序列为 $o_{t+1},\cdots,o_T$ 的后向概率 $\beta_t(i)$，只需考虑在时刻 t+1 所有可能的 N 个状态 $q_j$ 的转移概率（$\alpha_{ij}$），以及此状态下观测 $o_{t+1}$ 的观测概率（$b_j(o_{t+1})$），然后考虑状态 $q_j$ 之后的观测序列的后向概率 （$\beta_{t+1}(j)$）。

步骤 3 求 $P(O|\lambda)$ 的思路和步骤 2 一致，只是初始概率 $\pi_i$ 代替转移概率

利用前向和后向概率的定义可以将观测序列概率 $P(O|\lambda)$ 统一写成
$$
P(O|\lambda) = \sum_i\sum_j\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j),t\in[1,T-1]
$$

### 一些概率与期望值的计算

利用前向和后向概率，可以得到关于单个状态和两个状态概率的计算公式

1. 给定模型和观测序列，在时刻 t 处于状态 $q_i$ 的概率，记
   $$
   \gamma_t(i) = P(s_t=q_i|O,\lambda)=\frac{P(s_t=q_i,O|\lambda)}{P(O|\lambda)}
   $$
   可以通过前向后向概率计算

   由前向概率 $\alpha_t(i)$ 和后向概率 $\beta_t(i)$ 定义可知：
   $$
   \alpha_t(i)\beta_t(i) = P(s_t=q_i,O|\lambda)\\
   \rightarrow \gamma_t(i) = \frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)}=\frac{\alpha_t(i)\beta_t(i)}{\sum_j\alpha_t(j)\beta_t(j)}\tag{10.24}
   $$

2. 给定模型和观测序列，在时刻 t 处于状态 $q_i$ 且在时刻 t+1 处于状态 $q_j$ 的概率，记
   $$
   \xi_t(i,j) = P(s_t=q_i,s_{t+1}=q_j|O,\lambda)
   $$
   可以通过前向后向概率计算
   $$
   \xi_t(i,j) = \frac{P(s_t=q_i,s_{t+1}=q_j|O,\lambda)}{P(O|\lambda)}\\=\frac{P(s_t=q_i,s_{t+1}=q_j|O,\lambda)}{\sum_i\sum_jP(s_t=q_i,s_{t+1}=q_j|O,\lambda)}\\
   P(s_t=q_i,s_{t+1}=q_j|O,\lambda) = \alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)\\
   \rightarrow \xi_t(i,j) = \frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{\sum_k\sum_j\alpha_t(k)a_{kj}b_j(o_{t+1})\beta_{t+1}(j)}\tag{10.26}
   $$

3. 将 $\gamma_t(i)、\xi_t(i,j)$ 对各个时刻 t 求和，可以得到一些有用的期望值

   - 在观测序列 O 下状态 $q_i$ 出现的期望值
     $$
     \sum_{t\in[1,T]}\gamma_t(i)
     $$

   - 在观测序列 O 下由状态 $q_i$ 转移的期望值
     $$
     \sum_{t\in[1,T-1]}\gamma_t(i)
     $$

   - 在观测序列 O 下由状态 $q_i$ 转移到状态 $q_j$ 的期望值
     $$
     \sum_{t\in[1,T-1]}\xi_t(i,j)
     $$

# 学习算法

隐马尔可夫模型的学习，根据训练数据是包括观测序列和对应的状态序列还是只有观测序列，可以分布由监督学习与非监督学习实现。

## 监督学习方法

假设已给训练数据包含 L 个长度相同的观测序列及对应的状态序列 $\{(O_1,S_1),(O_2,S_2),\cdots,(O_L,S_L)\}$，那么可以利用极大似然估计法估计隐马尔可夫模型的参数，具体如下

1. 转移概率 $a_{ij}$

   设样本中时刻 t 处于状态 $q_i$ 时刻 t+1 转移到状态 $q_j$ 的频数为 $A_{ij}$，那么状态转移概率的估计是
   $$
   \hat a_{ij} = \frac{A_{ij}}{\sum_jA_{ij}},i\in[1,N],j\in[1,N]
   $$

2. 观测概率 $b_j(k)$ 的估计

   设样本中状态为 $q_j$ 并观测为 $v_k$ 的频数是 $B_{jk}$，那么状态为 $q_j$ 观测为 $v_k$ 的概率 $b_j(k)$ 的估计是
   $$
   \hat b_j(k) = \frac{B_{jk}}{\sum_kB_{jk}},j\in[1,N],k\in[1,M]
   $$

3. 初始状态概率 $\pi$ 的估计 $\hat \pi_i$ 为 T 个样本中初始状态为 $q_i$ 的频率

由于监督学习需要使用训练数据，而人工标注训练数据往往代价很高，又是就会利用非监督学习的方法

## Baum-welch

假设给定训练数据只包含 L 个长度为 T 的观测序列 $\{O_1,O_2,\cdots,O_L\}$ 而没有对应的状态序列，目标是学习隐马尔可夫模型 $\lambda = (A,B,\pi)$ 的参数。我们将观测序列数据看作观测数据 O，状态序列数据看作隐数据 S，那么隐马尔可夫模型事实上是一个含有隐变量的概率模型
$$
P(O|\lambda)=\sum_SP(O|S,\lambda)P(S|\lambda)
$$
它的参数学习可以由 EM 算法实现

1. 确定完全数据的对数似然函数

   所有观测数据写成 $O=(o_1,o_2,\cdots,o_T)$，所有隐数据写成 $S=(s_1,s_2,\cdots,s_T)$，完全数据则为 $(O,S) = (o_1,o_2,\cdots,o_T,s_1,s_2,\cdots,s_T)$。完全数据的对数似然函数是 $logP(O,S|\lambda)$。

2. EM 的 E 步：求 Q 函数 $Q(\lambda,\bar\lambda)$
   $$
   Q(\lambda,\bar\lambda) = \sum_S[logP(O,S|\lambda)|O,\bar\lambda]\\
   略去对 \lambda 而言的常数因子 1/P(O|\bar\lambda)\\
   =\sum_SlogP(O,S|\lambda)P(O,S|\bar\lambda)
   $$
   $\bar\lambda$ 为模型参数的当前估计值，$\lambda$ 是要极大化的隐马尔可夫模型参数
   $$
   P(O,S|\lambda) = \pi_{s_1}b_{s_1}(o_1)a_{s_1s_2}b_{s_2}(o_2)\cdots a_{s_{T-1}s_{T}}b_{s_T}(o_T)\\
   \downarrow\\
   Q(\lambda,\bar\lambda) = \sum_Slog\pi_{s_1}P(O,S|\bar\lambda)+\sum_S(\sum_{t\in[1,T-1]}loga_{s_ts_{t+1}})P(O,S|\bar\lambda) \\+ \sum_S(\sum_{t\in [1,T]}logb_{s_t}(o_t))P(O,S|\bar\lambda)
   $$

3. EM 的 M 步：极大化 Q 函数求模型参数

   由于要极大化的参数在上式中单独出现在三个项中，所以只需对各项分布极大化

   （1）第一项为
   $$
   \sum_Slog\pi_{s_1}P(O,S|\bar\lambda)=\sum_ilog\pi_iP(O,s_1=q_i|\bar\lambda)
   $$
   $\pi_i$ 满足约束条件 $\sum_i \pi_i = 1$，利用拉格朗日乘子法，得到拉格朗日函数：
   $$
   \sum_ilog\pi_iP(O,s_1=q_i|\bar\lambda) + \gamma(\sum_i\pi_i - 1)
   $$
   
   对 $\pi_i$ 求偏导得
   $$
   \frac\partial{\partial\pi_i}[\sum_ilog\pi_iP(O,s_1=q_i|\bar\lambda) + \gamma(\sum_i\pi_i - 1)] = 0\tag{10.35}
   $$
   得
   $$
   P(O,s_1=q_i|\bar\lambda)+\gamma\pi_i = 0
   $$
   对 i 求和得到 $\gamma$
   $$
   \gamma=-P(O|\bar\lambda)
   $$
   代入 10.35 得
   $$
   \pi_i = \frac{P(O,s_1=q_i|\bar\lambda)}{P(O|\bar\lambda)}\tag{10.36}
   $$
   （2）第二项为
   $$
   \sum_S(\sum_{t\in[1,T-1]}loga_{s_ts_{t+1}})P(O,S|\bar\lambda) = \\\sum_i\sum_j\sum_t loga_{ij}P(O,s_t=q_i,s_{t+1}=q_j|\bar\lambda)
   $$
   应用具有约束条件 $\sum_j a_{ij} = 1$ 的拉格朗日乘子法得到
   $$
   a_{ij} = \frac{\sum_{t\in[1,T-1]}P(O,s_t=q_i,s_{t+1}=q_j|\bar\lambda)}{\sum_{t\in[1,T-1]}}P(O,s_t=q_i|\bar\lambda)\tag{10.37}
   $$
   （3）第三项为
   $$
   \sum_S(\sum_{t\in [1,T]}logb_{s_t}(o_t))P(O,S|\bar\lambda) = \sum_j\sum_{t\in[1,T]}logb_{q_j}(o_t)P(O,s_t=q_j|\bar\lambda)
   $$
   约束条件为 $\sum_kb_{q_j}(v_k)=1$。只有 $o_t=v_k$ 时 $b_{q_j}(o_t)$ 的偏导数才不为 0，以 $I(o_t=v_k)$ 表示，求得
   $$
   b_{q_j}(v_k) = \frac{\sum_tP(O,s_t=q_j|\bar\lambda)I(o_t=v_k)}{\sum_tP(O,s_t=q_j|\bar\lambda)}\tag{10.38}
   $$

## Baum-Welch 模型参数估计

将 10.36 - 10.38 中的各概率分布用 $\gamma_t(i),\xi_t(i,j)$ 表示，则可将公式写成
$$
a_{ij} = \frac{\sum_t\xi_t(i,j)}{\sum_t\gamma_t(j)}\\
b_j(k) = \frac{\sum_{t,o_t=v_k}\gamma_t(j)}{\sum_t\gamma_t(j)}\\
\pi_i = \gamma_1(i)
$$
$\gamma_t(i),\xi_t(i,j)$ 分别由式 10.24，10.26 给出。

**算法 10.4**

1. 初始化

   对 n=0，选取 $a_{ij}^0,b_j(k)^0,\pi_i^0$，得到模型 $\lambda^o = (A^0,b^0,\pi^0)$

2. 递推，对 n=1，2，...，
   $$
   a_{ij}^{n+1} =\frac{\sum_t\xi_t(i,j)}{\sum_t\gamma_t(j)}\\
   b_j(k)^{n+1} = \frac{\sum_{t,o_t=v_k}\gamma_t(j)}{\sum_t\gamma_t(j)}\\
   \pi_i^{n+1} = \gamma_1(i)
   $$

3. 终止，得到模型参数 $\lambda^{n+1}$

# 预测算法

近似算法与维特比算法（Viterbi)

## 近似算法

在每个时刻 t 选择在该时刻最有可能出现的状态 $s_t^*$，从而得到一个状态序列 $S^* = (s_1^*,s_2^*,\cdots,s_T^*)$，将他作为预测的结果

给定隐马尔可夫模型 $\lambda$ 和观测序列 O，在时刻 t 处于状态 $q_i$ 的概率 $\gamma_t(i)$ 是
$$
\gamma_t(i) = \frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)} = \frac{\alpha_t(i)\beta_t(i)}{\sum_j\alpha_t(i)\beta_t(i)}
$$
在每一时刻 t 最有可能的状态 $s_t^*$ 是
$$
s_t^* = argmax_{i\in[1,N]}[\gamma_t(i)],t\in[1,T]
$$
从而得到状态序列

该方法优点是计算简单，缺点是不能保证预测的状态序列整体是最有可能的状态序列，因为预测的状态序列可能有实际不发生的部分，而且该方法得到的状态序列中有可能存在转移概率为 0 的相邻状态，即对某些 $i，j，a_{ij}=0$ 时。尽管如此，近似算法仍是有用的

## 维特比算法

维特比算法实际是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径（最优路径），这是一条路径对应着一个状态序列

根据动态规划原理，最优路径具有这样的特性：如果最优路径在时刻 t 通过节点 $s_t^*$，那么这一路径从节点 $s_t^*$ 到终点 $s_T^*$ 的部分路径，对于从 $s_t^*$ 到 $s_T^*$ 的所有可能的部分路径来说，必须是最优的。因此，只需从时刻 t=1 开始，递推地计算在时刻 t 状态为 $q_i$ 的各条部分路径的最大概率，直至得到时刻 t=T 状态为 $q_i$ 的各条路径的最大概率。时刻 t=T 的最大概率即为最优路径的概率 $P^*$，最优路径的终结点 $s_T^*$ 也同时得到。之后，为了找出最优路径的各个节点，从终结点 $s_T^*$ 开始，由后向前逐步求得节点 $s_{T-1}^*,\cdots,s_1^*$，得到最优路径 $S^*$。这就是维特比算法

首先导入两个变量 $\delta$ 和 $\Psi$。定义在时刻 t 状态为 $q_i$ 的所有单个路径 $(s_1,s_2,\cdots,s_t)$ 中概率最大值为
$$
\delta_t(i) = max_{s_1,s_2,\cdots,s_{t-1}}P(s_t=q_i,s_{t-1},\cdots,s_1,o_t,\cdots,o_1|\lambda),i\in[1,N]\\
\delta_{t+1}(j) = max_{s_1,s_2,\cdots,s_{t}}P(s_{t+1}=q_j,s_{t},\cdots,s_1,o_{t+1},\cdots,o_1|\lambda)\\
=max_{i\in[1,N]}[\delta_t(i)a_{ij}]b_j(o_{t+1}),i\in[1,N],t\in[1,T-1]\\
$$
定义在时刻 t 状态为 $q_i$ 的所有单个路径 $(s_1,s_2,\cdots,s_{t-1},s)$ 中概率最大的路径的第 t-1 个节点为
$$
\Psi_t(j) = argmax_{i\in[1,N]}[\delta_{t-1}(i)a_{ij}],i\in[1,N]
$$
**算法 10.5 维特比算法**

输入：模型 $\lambda$ ，观测序列 O

输出：最优路径 $I^*$

（1）初始化
$$
\delta_1(i) = \pi_ib_i(o_1)\\
\Psi_1(i) = 0,i\in[1,N]
$$
（2）递推，对 t = 2,3,...,T
$$
\delta_t(i) = max_j[\delta_{t-1}(j)a_{ji}]b_i(o_t)\\
\Psi_t(i) = argmax_j[\delta_{t-1}(j)a_{ji}]
$$
（3）终
$$
P^*=max_i\delta_T(i)\\
s_T^*=argmax_i\delta_T(i)
$$
（4）最优路径回溯，对 t = T-1,T-2,...,1
$$
s_t^* = \Psi_{t+1}(s_{T+1}^*)
$$
得到最优路径 $S^*$

例 利用例 2 的模型及观测序列，试求最优状态序列，即最优路径 $S^*$

（1）初始化，在 t = 1，对每个状态 $q_i$，i = 1,2,3，求状态为 $q_i$ 观测 $o_1$ 为红的概率，记此概率为 $\delta_1(i)$
$$
\delta_1(i) = \pi_ib_i(o_1)\\
\delta_1(1) = 0.10,\delta_1(2)=0.16,\delta_1(3)=0.28\\
\Psi_1(i)=0,i=1,2,3
$$
（2）在 t = 2 时，对每个状态 $q_j$，求在 t = 1 时状态为 $q_i$ 观测为红并在 t = 2 时状态为 $q_i$ 观测 $o_2$ 为白的路径的最大概率
$$
\delta_2(j)=max_i\delta_1(i)a_{ij}b_j(o_2)
$$
同时，对每个状态 $q_j$，记录概率最大路径的前一个状态 i
$$
\Psi_2(j) = argmax_i\delta_1(i)a_{ij}
$$
计算
$$
\delta_2(1) = max(0.01*0.5,0.16*0.3,0.28*0.2)*0.5=0.028\\
\Psi_2(1)=3\\
\delta_2(2) = 0.0504,\Psi_2(2) = 3\\
\delta_2(3) = 0.042,\Psi_2(3)=3
$$
同样方法计算 t = 3 时的值
$$
\delta_3(1) = 0.00756,\Psi_3(1) = 2\\
\delta_3(2) = 0.01008,\Psi_3(2) = 2\\
\delta_3(3) = 0.0147,\Psi_3(3) = 3\\
$$
（3）以 $P^*$ 表示最优路径的概率，则
$$
P^* = max_{i\in[1,3]}\delta_3(i) = 0.0147
$$
最优路径的终点是 $s_3^*$
$$
s_3^* = argmax_i\delta_3(i) = 3
$$
（4）有最优路径的终点 $s_3^*$，逆向找到 $s_2^*,s_1^*$
$$
t=2,s_2^*=\Psi_3(s_3^*)=3\\
t=1,s_1^*=\Psi_2(s_2^*)=3\\
$$
最优路径因此为 $S^* = (3,3,3)$