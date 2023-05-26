# 读相关论文
## NCF(Neural Collaborative Filtering)
**协同过滤**，基于用户过去与商品的互动（比如评价、点击等）来对商品的偏好进行建模。

对于协同过滤，有很多不同的技术，比如矩阵分解（MF）。

**矩阵分解**，将用户和商品投影到共享的潜在空间中，使用潜在的特征向量来表示用户或物品。所以，用户在一个物品上的交互（评价、点击等）就被建模为它们潜在向量的内积。缺点：这种内积可能不足以捕获用户交互数据的复杂结构。

这篇文章探索使用DNNs来从数据中学习交互函数，相比较MF方法，使用DNNs进行推荐所做的工作将更少。

他们关注的是**隐式反馈**，就是那种能间接反映用户偏好的行为（比如观看视频、购买商品、点击物品等）。因为相比**显示反馈**（评价、评级等），**隐式反馈**可以自动追踪，对于内容提供者来说较容易收集。但是，这种方式因为没有观察到用户的满意度，所以**负面反馈**是稀缺的。

所以他们在这个文章中也探讨了利用DNNs来建模**噪声隐式反馈信号**这个中心主题。

并且他们的主要工作如下：
![picture 1](assets/images/1682000366388.png)

### 2. 准备工作
#### 2.1 从隐式数据中学习
假设有$M$个用户，$N$个商品。然后定义用户商品交互矩阵$Y$（吴恩达课程中讲的二进制标签）。
> ![picture 2](assets/images/1682034787699.png)  

这里有个问题：如果$y_{ui}=1$，不能说明用户$u$喜欢商品$i$；同理，如果$y_{ui}=0$，也不能说明不喜欢，可能只是用户不知道。所以，矩阵$Y$只提供了关于用户偏好的嘈杂信号。

隐式反馈的推荐问题表述为估计$Y$中未观察条目的分数问题，这些分数用于对条目进行排序。

$\hat y_{ui} = f(u,i|Θ)$，$\hat y_{ui}$表示交互$y_{ui}$预测的分数，$Θ$表示模型参数，$f$表示模型参数到预测分数的映射函数，

为了估计参数$Θ$，现有的方法通常遵循优化目标函数的机器学习范式。文献中常用的目标函数有两种，分别是**点损失**和**成对损失**。

**点损失**，**点学习**，通常遵循回归框架，最小化$\hat y_{ui}$与目标值$y_{ui}$之间的平方损失。对于未观察条目，一种是将其直接视为负反馈，第二种是从未观察条目中采样负反馈实例。

**成对损失**，**成对学习**，最大化观察到的条目$\hat y_{ui}$与未观察到的条目$\hat y_{ui}$之间的余量。观察条目排名应高于未观察条目。

对于**NCF**，使用神经网络将交互函数$f$参数化，来估计$\hat y_{ui}$。因此自然支持**点学习**和**成对学习**。

#### 2.2 MF(Matrix Factorization)矩阵分解
定义了两个潜在向量$\vec{p}_u$和$\vec{q}_i$，分别表示用户$u$和物品$i$。矩阵分解使用$\vec{p}_u$和$\vec{q}_i$的**内积**来估计交互值$y_{ui}$。
> ![picture 3](assets/images/1682043424866.png) 

其中，$K$表示潜在空间的**维度**。

由此可以看出，MF建模了用户和物品潜在因素的双向交互（<u>我的理解是因为内积，二者都影响最终结果</u>），并且假设潜在空间的各个维度之间相互独立（<u>各内积各的，不影响</u>）且权重相同（<u>每一项系数都是1</u>），然后线性组合（<u>每一项相加起来</u>）。所以，MF是潜在因素的线性模型。

使用下图做了举例，说明了内积函数是如何限制MF的表达性的。
> ![picture 4](assets/images/1682045242701.png)  

为了更好地理解这个例子，有两个设置要说明一下：
1. 由于MF将用户和物品映射到相同的潜在空间，所以判断两个用户的相似性也是可以用**内积**，或者另一种等价做法是利用两个用户的潜在向量的**夹角的余弦值**（当然，这里假设两个向量都是**单位向量**）。
2. 不失一般性，我们使用**Jaccard coefficient**作为MF需要恢复的两个用户的**真值相似度**。

> **Jaccard coefficient**
> 设$R_u$为用户$u$已经交互过的物品的集合（<u>就是那些$y_{ui}=1$的</u>），然后两个用户$i$和$j$的**Jaccard 相似性**就可以表示为$s_{ij}=\frac{\mid R_i\mid\bigcap\mid R_j\mid}{\mid R_i\mid\bigcup\mid R_j\mid}$，结果是个小数。

根据Figure 1a，可以得出$s_{23}=0.66>s_{12}=0.5>s_{13}=0.4$，故三个用户对应向量$\vec{p}_1 \vec{p}_2 \vec{p}_3$**必然**为Figure 1b的情况（<u>因为$s_{ij}$是相似性</u>）。

加入第四个用户$u_4$，我们可以得到相似性$s_{41}=0.6>s_{43}=0.4>s_{42}=0.2$，说明$u_4$与$u_1$最相似，但无论如何放置$\vec{p}_4$（在先保证与$\vec{p}_1$夹角余弦值最小的情况下）都是距离$\vec{p}_2$更近，而不是$\vec{p}_3$。

上述就是使用固定简单的内积的局限性，当然解决上述问题可以使用大量的$K$，但是又会不利于模型的泛化。所以下一小节将使用DNNs来解决这个限制。

### 3. NCF(NEURAL COLLABORATIVE FILTERING)
#### 3.1 总体框架
使用两个特征向量表示用户和物品，作为输入层，是稀疏的。然后上面一层是嵌入层，将输入层向量化，是全连接层。再往上是神经协同过滤层。最终输出层是预测分数，并通过最小化逐点损失来训练。
![picture 5](assets/images/1682067176999.png)  

将NCF预测模型表述为下图：
![picture 6](assets/images/1682079522685.png)  

由于NCF层是多层，故又表示成：
![picture 7](assets/images/1682079569143.png)  

由于需要将输出的范围控制在$[0,1]$之间，所以激活函数使用了$sigmoid$。

然后，定义了似然函数：
![picture 8](assets/images/1682079813334.png)  

并取似然函数的负对数，得到
![picture 9](assets/images/1682079851402.png)  

这就是NCF要最小化的目标函数，它的优化可以通过随机梯度下降完成。（因为上文提到了平方损失与隐式数据不太吻合，所以就用了上面这种二值交叉熵loss）

#### 3.2 GMF(Generalized Matrix Factorization)
证明MF是NCF的一个特例。

**用户潜在向量**就是$P^T\vec v^U_u$，**物品潜在向量**就是$Q^T\vec v^I_i$，二者作为嵌入层。然后定义NCF层的第一层的映射函数为
![picture 10](assets/images/1682083729292.png)  

$\odot$表示两个向量逐个元素乘积。

然后将这个结果向量投影到输出层，得到
![picture 11](assets/images/1682083989586.png)  

$a_{out}$表示输出层激活函数，$\vec h$表示边权。

<u>理解：就是**用户**和**物品**两个向量作元素相乘，然后再与**边权向量**作点积，然后再加上**激活函数**，使结果维持在[0,1]。</u>

确实，如果**边权向量**的元素都为1，**激活函数**使之输出值不变，看起来就是**MF**。

综上，实现了一个广义矩阵分解，**激活函数**为$sigmoid$，向量$\vec h$使用**二值交叉熵损失**从数据中学习。

#### 3.3 MLP(Multi-Layer Perceptron)
用户和物品向量进行拼接，而不是对应元素相乘。激活函数使用$ReLU$。网络结构使用塔式，底层最宽，往上依次减半。

#### 3.4 Fusion of GMF and MLP
到此为止已经用NCF开发了两个实例，一个是运用线性核心对潜在特征交互建模的GMF，另一个是使用非线性从数据中学习交互函数的MLP。为了使二者相互增强，并且能为更复杂的用户交互建模，来融合二者。

一种最直接的方式是让GMF和MLP共享相同的嵌入层，然后再组合二者的交互函数作为输出。具体来说，GMF与单层MLP相结合的模型可以表示为
![picture 12](assets/images/1682122426637.png)  

但是，这种方式会限制融合层模型的性能，比如二者需要使用相同大小的嵌入层，但二者的最优嵌入层可能不一样，所以灵活性不好。

为了更具灵活性，让二者使用不同的嵌入层，连接它们最后一个隐藏层来组合两个模型。图和公式如下
![picture 13](assets/images/1682122958707.png)  

![picture 14](assets/images/1682122972816.png)  

其中，$\vec p^G_u$和$\vec p^M_u$分别表示GMF和MLP的用户嵌入层，同理，$\vec q^G_i$和$\vec q^M_i$表示物品的。同上文，MLP层的激活函数使用$ReLU$。最后，称这个模型为"NeuMF(Neural Matrix Factorization)"。模型中关于每个参数的导数都可以通过反向传播计算。

##### 3.4.1 预训练
文献中推荐使用预训练，事先训练GMF和MLP的模型直到收敛，然后将二者的模型参数作为NeuMF的相应部分参数的初始化。唯一的调整是在输出层，将两个模型的参数作连接操作：
![picture 15](assets/images/1682128305667.png)  

其中，$α$表示决定两个预训练模型之间权重的超参数（~~在代码中好像只做了连接操作，没有加入这个超参数~~。代码中$α$是0.5，在下文实验小节也提到了）。

对于训练GMF和MLP，采用了自适应矩估计，它通过对频繁的参数执行较小的更新和对不频繁的参数执行较大的更新来适应每个参数的学习率。在将预训练的参数输入NeuMF后，我们使用朴素SGD而不是Adam来优化它。（<u>但是，在代码中并没有见到使用SGD，通过对代码的debug，发现训练之前，优化器还是adam</u>）

### 4. EXPERIMENTS
实验的目的是为了解读下面三个问题：
1. 我们提出的NCF方法是否优于最先进的隐式协同过滤方法?
2. 我们提出的优化框架(负采样的对数损失)如何用于推荐任务?
3. 更深层次的隐藏单元是否有助于从用户项目交互数据中学习?

#### 4.1 实验设置
**数据集**选择了两个，分别是**MovieLens**和**Pinterest**。

**MovieLens**是一个显示反馈数据集，在本项目中，转为了隐式反馈数据集（即每个条目被标记为0或1）。

**评价协议**，文献中的评价方法采用留一法交叉验证，性能指标使用HR(Hit Ratio)和NDCG(Normalized Discounted Cumulative Gain)。

> <a href='https://zhuanlan.zhihu.com/p/493958358'>HR和NDCG的知乎文章</a>
> 
> **HR**，强调的是模型推荐的准确性，即用户的需求项是否包含在模型的推荐项中。
> **NDCG**，强调的是用户的需求项在模型推荐列表中的位置，越靠前越佳。

**Baselines**，选出4个模型与NCF的实例（GMF、MLP和NeuMF）比较。分别是：**ItemPop**、**ItemKNN**、**BPR**和**eALS**。

**参数设置**，损失函数均使用**log loss**，每个正实例均增加四个负实例，初始化模型参数使用**高斯分布**（均值为0，标准差为0.01），**batch_size**测试了[128, 256, 512, 1024]，**学习率**测试了[0.0001, 0.0005, 0.001, 0.005]，**预测因子**测试了[8, 16, 32, 64]，MLP的隐含层有3层。

文献剩下的内容都在讲图了就。

### 总结
通过这篇文章，我了解了推荐系统其中的一个方向：协同过滤。在这篇文献中，作者将神经网络与协同过滤相结合，提出了神经协同过滤框架，来处理基于隐式反馈的推荐。对比传统的方法(如MF和MF的推广等算法)，NCF(文中实现了三个实例，包括GMF、MLP和NeuMF)能够提供更好的推荐性能。

#### 1. 学习过程
##### 1.1 准备工作
文献阅读，下载知云，创建md笔记。

代码，克隆代码，创建conda环境，根据README.md调试运行环境。

##### 1.2 阅读论文
阅读论文方式，一句一句阅读，先自己翻译，再看知云翻译，见到专业名称就查找相关资料(文字或视频)学习，边读论文边做笔记。

##### 1.3 代码调试运行
阅读到该文献的第三节(3.2, 3.3, 3.4)，开始结合论文讲述的模型框架看具体代码实现，过程中不断利用debug查看一些变量的取值和方法的参数。

值得一提的是，在调试代码的过程中发现了一个问题: 在使用预训练的NeuMF中，代码中，优化器并没有切换到SGD。然后我就想看一下Adam和SGD的具体效果区别，但是由于数据集太大，无法快速跑出模型，我就"随机"切出了子数据集，开始了训练。(其实我感觉人作者可能只是代码没提交，或者这个地方影响不大😂)

经过上述一番"折腾"，算是成功地把代码摸熟了。

#### 2. 疑问
问题1: 这个预测因素(the number of predictive factors，代码中是num_factors，用于Embedding时的output_dim了)到底是什么？

问题2: 在使用预训练的NeuMF中，代码中，优化器并没有切换到SGD(论文中在3.4.1节说要在预训练模型中使用SGD)。
![picture 17](assets/images/1682237389170.png)

裁剪数据集后，下图分别是GMF、MLP、NeuMF(Adam)和NeuMF(SGD)：
![picture 19](assets/images/1682322742407.png)  
![picture 20](assets/images/1682322747129.png)  
![picture 21](assets/images/1682322760186.png)  
![picture 22](assets/images/1682322766085.png)  

对于NeuMF，优化器从Adam改为SGD后，训练速度有所提升，但是HR和NDCG下降了。

#### 3. 感悟
作为第一篇文献，一上来最直接的感受就是"哇，好多专业名词"，所以在读论文的过程中不断地查阅和学习相关知识，这也让我对这个方向有了一些了解。

通过读论文，我发现自己的机器学习相关基础知识的储备还不够，所以我接下来要加快学习的步伐，打好基础。

#### 4. 附录
论文的每一章的小结。对于第1节INTRODUCTION，了解了本文要做的工作，初步了解了矩阵分解MF。对于第2节PRELIMINARIES，定义了用户-物品交互矩阵，通过举例的方式，展示了MF的简单内积的局限性。对于第3节NCF，主要介绍了NCF的总体框架，还证明了MF是NCF的一个特例，并且实现了三个NCF实例(分别是GMF，MLP和NeuMF)。对于第4节EXPERIMENTS，文中先提出了试验阶段要回答的三个问题，并说明了实验的设置(包括数据集，评价方法，Baselines和实验参数)，然后针对每个问题做了相关实验。对于第5节是相关工作。第6节是总结和展望。

## FREEDOM(Freezing and Denoising Graph Structures for Multimodal Recommendation)
认为LATTICE的潜在图学习结构是低效且不必要的。实验证明，在训练之前冻结物品-物品的结构也可以达到相匹敌的性能。基于这一发现，提出了FREEDOM(FREEzes the item-item graph and DenOises the user-item interaction graph simultaneously for Multimodal recommendation)。

> Compared with LATTICE, FREEDOM achieves an average improvement of 19.07% in recommendation accuracy while reducing its memory cost up to 6× on large graphs.

### 模型框架
模型共分为四个部分：1.Constructing Frozen Item-Item Graph；2.Denoising User-Item Bipartite Graph；3.Integration of Two Graphs for Learning；4.Top-K Recommendation。

#### 1.Constructing Frozen Item-Item Graph $S-Frozen$
FREEDOM也是对每个**模态**$m$的**原始特征**使用**kNN**构建**初始模态感知“物品-物品”图**$S^m$。

1. 首先，$N$个物品，对初始特征$x_i^m$和$x_j^m$计算**余弦相似性**，如**公式(1)**$S^m_{ij}=\frac{(x^m_i)^Tx^m_j}{||x^m_i||||x^m_j||}$，其中$S^m_{ij}$是矩阵$S^m\in R^{N\times N}$的第$i$行，第$j$列的元素。
2. 其次，使用kNN稀疏化，并将带权重的矩阵$S^m$转为无权重矩阵$\hat S^m$，如**公式(2)**$\hat S^m_{ij}=\begin{cases}
    1 & S^m_{ij}∈topk(S^m_i) \\
    0 & otherwise
\end{cases}$，其中，将$1$定义为两个物品$i$和$j$有**潜在联系**，需要注意是$\hat S^m$不同于**LATTICE**中带权重的相似性矩阵。
3. 然后，同样地对矩阵$\hat S^m$归一化，得到矩阵$\widetilde{S}^m=(D^m)^{-\frac{1}{2}}\hat{S}^m(D^m)^{-\frac{1}{2}}$。
4. 再然后，对每个模态的$\widetilde{S}^m$进行整合，如**公式(3)**$S=\sum\limits_{m\in M}\alpha_m\widetilde{S}^m$，其中，$\alpha_m$表示**模态**$m$的**重要性评分**，$M$是**模态集**。在此，定义模态集$M=\{v,t\}$，定义超参数$\alpha_v$表示视觉模态的重要性，并令$\alpha_t=1-\alpha_v$。
5. 最后，**冻结潜在物品-物品图**$S$得到$S-Frozen$。

#### 2.Denoising User-Item Bipartite Graph

## LATTICE(Mining Latent Structures for Multimedia Recommendation)
先前的工作是使用**多模态特征**作为**副信息**来对**“用户-物品”交互**建模，但是这种方式不适合推荐系统。具体来说，只是通过**高阶“物品-用户-物品”关系**来隐式地建模**协同“物品-物品”关系**。

> The majority of previous work focuses on modeling **user-item interactions** with **multimodal features** included as **side information**. However, this scheme is not well-designed for multimedia recommendation. Specifically, only **collaborative item-item relationships** are implicitly modeled through **high-order item-user-item relations**.

### 模型框架
$U$和$I$为用户集和物品集，$u$表示一个用户，$u∈U$。如果用户$u$与物品$i$有关联，那么说有正反馈，用$y_{ui}=1$表示，其中$i∈I^u$。用$x_u,x_i∈R^d$表示某个用户和物品的输入ID嵌入，其中$d$表示嵌入层的维度。用$e^m_i∈R^{d_m}$表示物品$i$的某个模态的特征，其中$d_m$表示某个模特下的特征的维度，用$m∈M$表示模态，$M$表示模态集。

模型共分为3个部分：1.Modality-aware Latent Structure Learning，2.Graph Convolutions和3.Combining with Collaborative Filtering。如下图。
![picture 1](assets/images/1685004655739.png)

#### 1. Modality-aware Latent Structure Learning
##### 第一步，Constructing initial 𝑘NN modality-aware graphs $\widetilde{S}^m$
1. 首先，对于$e^m_i$，计算与另一特征$e^m_j$之间的**余弦相似度**，得到图邻接矩阵$S^m_{ij}$，如下**公式(1)**$S^m_{ij}=\frac{(e^m_i)^Te^m_j}{||e^m_i||||e^m_j||}$。
2. 其次，控制邻接矩阵的元素非负，范围为$[0,1]$。
3. 然后，利用`kNN`稀疏化邻接矩阵，如下**公式(2)**$\hat{S^m_{ij}}=\begin{cases}
    S^m_{ij} & S^m_{ij}∈topk(S^m_i) \\
    0 & otherwise
\end{cases}$，其中结果$\hat{S^m}$是**稀疏化的有向图邻接矩阵**。
4. 最后，对$\hat{S}^m$进行归一化，如下**公式(3)**$\widetilde{S}^m=(D^m)^{-\frac{1}{2}}\hat{S}^m(D^m)^{-\frac{1}{2}}$，其中$D^m$表示$\hat{S}^m$的**对角度矩阵**，其计算方法为$D^m_{ii}=\sum_{j}\hat{S}^m_{ij}$，由此可以看出$D^m$只有对角线元素，元素值为$\hat{S}^m$的第$i$行元素之和。

##### 第二步，Learning latent structures $A^m$
1. 首先，对于$e^m_i$，计算**high-level feature vector**，如下**公式(4)**$\widetilde{e}^m_i=W_me^m_i+b_m$，其中，$W_m \in R^{d'\times d_m}$定义为**trainable transformation matrix**，$b_m \in R^{d'}$。
2. 然后，对$\widetilde{e}^m_i$根据公式(1)(2)(3)计算出邻接矩阵$\widetilde{A}^m$。
3. 最后，根据**公式(5)**$A^m=\lambda \widetilde{S}^m+(1-\lambda) \widetilde{A}^m$，得出最终的**图邻接矩阵**，其中，系数$\lambda \in (0,1)$，$\widetilde{S}^m$为上一步最后的计算结果。

##### 第三步，Aggregating multimodal latent graphs 得到 $A$
1. 首先，利用**公式(6)**$A=\sum_{m=0}^{|M|}a_mA^m$计算出最终的**Latent Structure**$A$，其中，$a_m$是模态$m$的重要度评分，$A$是表示多个模态的物品关系的图结构。
2. 最后，使用**softmax function**让$A$归一化，并使$\sum_{m=0}^{|M|}a_m=1$。

#### 2. Graph Convolutions
利用**公式(7)**$h_i^{(l)}=\sum \limits_{j\in N(i)}A_{ij}h_j^{(l-1)}$进行图卷积，其中$N(i)$表示物品$i$的邻接物品，$h_i^{(l)}\in R^d$表示第$l$层的物品$i$的表示。

> We set the input item representation $h_i^{(0)}$ as its corresponding ID embedding vector $x_i$.
> After stacking $L$ layers, $h_i^{(L)}$ encodes the high-order item-item relationships that are constructed by multimodal information and thus can benefit the downstream CF methods.

#### 3. Combining with Collaborative Filtering
在任何CF方法之前使用以上步骤。

将$\widetilde{x}_u,\widetilde{x}_i\in R^d$定义为CF方法的用户和物品嵌入的输出。然后，通过增加从物品图中学习到的归一化的物品嵌入$h_i^{(L)}$增强物品嵌入，如**公式(8)**$\hat x_i=\widetilde{x}_i+\frac{h_i^{(L)}}{||h_i^{(L)}||_2}$。最后，计算**用户-物品偏好得分**(the user-item preference score)，如**公式(9)**$\hat y_{ui}=\widetilde x_u^T \hat x_i$。

#### 4. Optimization
使用**Bayesian Personalized Ranking (BPR) loss**。

### 实验
实验回答以下三个问题：
- How does our model perform compared with the state-of-the-art multimedia recommendation methods and other CF methods in both warm-start and cold-start settings?
- How effective are the item graph structures learned from multimodal features?
- How sensitive is our model under the perturbation of several key hyper-parameters?

#### 实验设置
**数据集**使用**Clothing**, **Sports**, and **Baby**。**模态**有**visual**和**textual**两种。
![picture 4](assets/images/1685063920356.png)  

#### Baselines
- MF
- NGCF
- LightGCN
- VBPR
- MMGCN
- GRCN
  
#### 实验结果
![picture 5](assets/images/1685063957784.png)  
***
![picture 6](assets/images/1685063970282.png) 
***
![picture 7](assets/images/1685063990607.png)
***
![picture 8](assets/images/1685064001159.png)  

### 结论
> In this paper, we have proposed **the latent structure mining method**(LATTICE) for multimodal recommendation, which leverages **graph structure learning** to discover **latent item relationships** underlying multimodal features. In particular, we first devise **a modality-aware graph structure learning layer** that learns item graph structures from multimodal features and fuses multimodal graphs. Along **the learned graph structures**, one item can receive **informative high-order affinities** from **its neighbors** by **graph convolutions**. Finally, we combine **our model** with **downstream CF methods** to make recommendations. Empirical results on three public datasets demonstrate the effectiveness of our proposed model.