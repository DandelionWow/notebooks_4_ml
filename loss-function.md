# Loss Functions总结
对不同领域（不同场景）的损失函数的总结。

## image-text(或text-image)匹配
在图文匹配场景下，损失函数可大致分为三类：对比损失、交叉损失和对抗损失。

- **对比损失**：这类损失函数的目的是让图像和文本之间的相似度高于图像和其他文本之间的相似度，或者让文本和图像之间的相似度高于文本和其他图像之间的相似度。它们通常使用正负样本对或者三元组作为输入，并要求正样本对之间的相似度大于负样本对之间的相似度加上一个边距。这类损失函数包括**Pairwise Contrastive Loss**，**Triplet Contrastive Loss**，**Ranking Loss**等。
- **交叉损失**：这类损失函数的目的是让图像和文本之间的相关性高于图像和其他文本之间的相关性，或者让文本和图像之间的相关性高于文本和其他图像之间的相关性。它们通常使用单个样本作为输入，并将其相关性与真实相关性进行比较。这类损失函数包括**Cross-Entropy Loss**，**KL-based Loss**等。
- **对抗损失**：这类损失函数的目的是让生成的图像或者文本与真实的图像或者文本难以区分，或者让生成的图像或者文本与真实的图像或者文本具有高度的一致性。它们通常使用一个判别器来区分真实数据和生成数据，并给出一个二分类损失。这类损失函数包括**Adversarial Loss**，**Adversarial Consistency Loss**等。

### Ranking Loss
Ranking Loss是一种用于衡量两个或多个样本之间的相对顺序或相似度的损失函数，它可以用于图文匹配的场景下，比如用于度量图像和文本之间的相似度。

Ranking Loss是一种基于排序的损失函数，是因为它的目标是使得匹配的样本之间的相似度或者距离排在不匹配的样本之间的相似度或者距离之前，也就是说，它要求模型对样本进行一个正确的排序。例如，对于图文匹配的任务，Ranking Loss要求给定一个图像，它与匹配的文本之间的相似度要高于它与不匹配的文本之间的相似度，也就是说，它要求模型对文本进行一个按照与图像相似度降序的排序。

Ranking Loss的一般形式是：

$$L=\sum_ {i=1}^N\max (0,\alpha -s (x_i, y_i)+s (x_i, y_j))$$

其中，$N$是一个批次中的样本数，$x_i$和$y_i$是第$i$个图像和文本对，$y_j$是第$i$个图像和其他不匹配的文本对，$s(\cdot, \cdot)$是一个相似度函数，比如余弦相似度或者欧氏距离，$\alpha$是一个边界参数。

这个损失函数的含义是，对于每个图像和文本对$(x_i, y_i)$，要使得它们之间的相似度大于它们与其他不匹配的文本对$(x_i, y_j)$之间的相似度，同时也要超过一个边界值。这样可以使得模型更加关注那些难以区分的负样本，从而提高图文嵌入模型的鲁棒性和泛化能力。

在Pytorch中，有一个内置的Ranking Loss函数，叫做MarginRankingLoss，它的定义如下：

``` python
class torch.nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```

它接受两个输入$x_1$和$x_2$，以及一个标签$y$（取值为1或-1），计算以下损失：

$$L=\max (0,-y * (x_1 - x_2) + \text {margin})$$

这个损失函数与上面的一般形式类似，只是使用了欧氏距离作为相似度函数，并且没有求和。它可以通过设置不同的参数来控制输出的形式，比如平均或者求和。它的使用示例如下：
``` python
import torch
import torch.nn as nn
# reduction设为none便于查看每个位置损失计算的结果
rankloss = nn.MarginRankingLoss(reduction='none')
x1 = torch.randn(10)
x2 = torch.randn(10)
# 随机生成10个0,1数据
y = torch.randint(0, 2, [10])
# 将y中数据为0的位置赋值为-1
y[y==0] = -1
loss = rankloss(x1, x2, y)
print(x1)
print(x2)
print(y)
print(loss)
```

输出

``` python
# x1
tensor([-0.1548,  0.1426, -0.1539, -0.0578,  0.0504, -0.3337, -0.0565,  0.0113,
        -0.0869,  0.1934])
# x2
tensor([ 0.1836, -0.0975, -0.0247, -0.0073, -0.0405,  0.0328, -0.0786,  1.0513,
        -1.0179, -1.0999])
# y
tensor([ 1, -1,  1, -1,  1, -1,  1, -1,  1, -1])
# 对应位置的损失计算结果
tensor([3.3804e-01, 2.4008e-01, 1.2923e-01, 5.0504e-02, 9.0897e-02, 3.6654e-01,
        2.2108e-02, 1.0400e+00, 9.3107e-01, 1.2933e+00])
```

### Triplet Loss
Triplet Loss是一种用于度量学习的损失函数，它的目的是使得同一类别的样本之间的距离小于不同类别的样本之间的距离，同时保持一定的间隔。Triplet Loss需要输入三个样本，分别是锚点（anchor）、正样本（positive）和负样本（negative）。锚点和正样本属于同一类别，而锚点和负样本属于不同类别。Triplet Loss的公式如下：

$$L = \max(d(a,p) - d(a,n) + margin, 0)$$

其中，$d(a,p)$表示锚点和正样本之间的距离，$d(a,n)$表示锚点和负样本之间的距离，$margin$表示一个超参数，用于控制两种距离之间的最小差值。Triplet Loss的目标是使得$d(a,p)$接近0，而$d(a,n)$大于$d(a,p) + margin$。

Pytorch中提供了一个内置的类`torch.nn.TripletMarginLoss`，可以直接用于计算Triplet Loss。这个类的构造函数如下：

```python
class torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
```

其中，`margin`是超参数，表示正负样本之间的最小间隔；`p`是距离度量的范数，默认为2，即欧氏距离；`eps`是一个小的正数，用于避免除以零；`swap`是一个布尔值，表示是否使用距离交换，即在计算损失时取正负样本中距离锚点更远的那个；`size_average`、`reduce`和`reduction`是用于控制损失函数的输出形式，分别表示是否对每个样本的损失求平均、是否对每个批次的损失求和或平均、以及具体的输出模式（'none' | 'mean' | 'sum'）。

这个类的前向传播函数如下：

```python
def forward(self, anchor, positive, negative)
```

其中，`anchor`、`positive`和`negative`都是输入张量，表示锚点、正样本和负样本。它们的形状应该是`(N, D)`或`(D)`，其中`N`是批次大小，`D`是特征维度。这个函数会返回一个张量，表示每个批次或每个样本的损失值。

使用这个类的示例代码如下：

```python
import torch
import torch.nn as nn
# 创建一个三元组损失函数对象
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
# 创建一些随机输入张量
anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)
# 计算输出损失
output = triplet_loss(anchor, positive, negative)
# 反向传播梯度
output.backward()
```

#### Multi-Modality Cross Attention Network for Image and Sentence Matching [LINK](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.pdf)
《基于交叉注意力机制的图像文本匹配》，这篇论文使用了Triplet Loss作为图像和文本之间的匹配损失函数。论文中给出了Triplet Loss的计算过程如下：

- 首先，对于每个图像-文本对$(I,S)$，使用一个多模态交叉注意力网络（MMCA）来提取它们在联合特征空间中的嵌入向量$(i,s)$。
- 然后，对于每个图像-文本对$(I,S)$，从训练集中随机选择一个与$I$相同类别但与$S$不同类别的文本$S_p$作为正样本，以及一个与$I$不同类别但与$S$相同类别的图像$I_n$作为负样本。
- 接着，使用MMCA网络分别提取$(I,S_p)$和$(I_n,S)$在联合特征空间中的嵌入向量$(i,s_p)$和$(i_n,s)$。
- 最后，计算Triplet Loss如下：

$$L_{tri} = \max(d(i,s) - d(i,s_p) + margin, 0) + \max(d(i,s) - d(i_n,s) + margin, 0)$$

其中，$d(i,s)$表示图像和文本嵌入向量之间的欧氏距离。

### Contrastive Loss
Contrastive Loss是一种基于二元组的损失函数，它要求两个相似的样本之间的距离要小于两个不相似的样本之间的距离，同时也加入一个边界来增加难度。它可以用于图文匹配的场景下，比如用于度量图像和文本之间的相似度。

图文匹配的场景下，Contrastive Loss的公式是：

$$L=\sum_ {i=1}^N\max (0,\alpha -s (I_i, T_i)+\max_ {j \neq i}s (I_i, T_j))+\max (0,\alpha -s (I_i, T_i)+\max_ {j \neq i}s (I_j, T_i))$$

其中，$N$是一个批次中的样本数，$I_i$和$T_i$是第$i$个图像和文本对，$s(\cdot, \cdot)$是一个余弦相似度函数，$\alpha$是一个边界参数。

这个公式的含义是，对于每个图像和文本对$(I_i, T_i)$，要使得它们之间的相似度大于它们与其他不匹配的图像或文本之间的相似度，同时也要超过一个边界值。这样可以使得模型更加关注那些难以区分的负样本，从而提高图文嵌入模型的鲁棒性和泛化能力。

#### VSE++: Improving Visual-Semantic Embeddings with Hard Negatives [LINK](https://arxiv.org/abs/1707.05612)
一个使用到Contrastive Loss的图文匹配的论文举例是《VSE++: Improving Visual-Semantic Embeddings with Hard Negatives》²，这篇论文是在BMVC 2018上发表的，它使用了Contrastive Loss来训练一个图文嵌入模型，比如从图像检索文本，或者从文本检索图像。

这篇论文的主要贡献是提出了一种使用hard negatives的方法，来提高图文嵌入模型的性能。(就是提出了上述损失函数)

### KL-based Loss
KL-based Loss是一种基于KL散度（Kullback-Leibler Divergence）的损失函数，它用来度量两个概率分布之间的相似度或距离。

KL散度的定义如下：

$$D_{KL}(P||Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}$$

其中，$P$和$Q$是两个离散概率分布，$i$是可能的取值。

KL散度的直观含义是，如果我们用分布$Q$来近似数据的真实分布$P$，那么我们需要付出的额外的信息量或编码损失。

在pytorch中，有一个内置的函数torch.nn.KLDivLoss()可以用来计算KL-based Loss。

这个函数接收三个参数，第一个是input，表示预测分布，第二个是target，表示真实分布，第三个是reduction，表示损失的聚合方式。

这里有一些注意事项：

- input和target必须有相同的形状。
- input必须在log空间中，即对预测概率取对数。
- target可以在log空间中或原始空间中，取决于log_target参数的值。
- reduction可以是"mean"（默认），"sum"或"none"，分别表示对损失取平均值，求和或不聚合。

举个例子，假设我们有一个三分类问题，预测分布和真实分布如下：

```python
import torch
import torch.nn as nn

input = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]])
target = torch.tensor([[0, 1, 0], [0, 0, 1]])
```

我们可以用torch.nn.KLDivLoss()来计算KL-based Loss：

```python
loss_fn = nn.KLDivLoss(reduction="mean")
loss = loss_fn(input.log(), target)
print(loss)
```

输出：

```python
tensor(0.5978)
```

#### Similarity Reasoning and Filtration for Image-Text Matching [LINK](https://arxiv.org/abs/2101.01368v1)
这篇论文的主要贡献是提出了一个基于相似性图推理和注意力过滤的图文匹配网络，其中使用了KL散度来度量投影兼容分布和标准化匹配分布之间的差异。

具体来说，这篇论文首先学习了基于向量的相似性表示，来综合地表征图像-句子之间的**全局对齐**和区域-单词之间的**局部对齐**。然后，引入了一个基于图卷积神经网络的相似性图推理（SGR）模块，来利用**局部和全局对齐之间的关系**，推断出**关系感知的相似性**。最后，设计了一个相似性注意力过滤（SAF）模块，来有效地整合这些对齐，通过选择性地关注有意义和有代表性的对齐，同时排除无意义的对齐的干扰。

在这个过程中，KL散度被用来计算**投影兼容分布**和**标准化匹配分布**之间的损失函数（就是CMPM损失函数，在[《Deep Cross-Modal Projection Learning for Image-Text Matching》](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ying_Zhang_Deep_Cross-Modal_Projection_ECCV_2018_paper.pdf)中提出）。**投影兼容分布**是指图像-句子或区域-单词之间的余弦相似性经过softmax归一化后得到的概率分布。**标准化匹配分布**是指图像-句子或区域-单词是否匹配的二值标签经过归一化后得到的概率分布。

KL散度损失函数的公式如下：

$$L_{KL} = \sum_{i=1}^N \sum_{j=1}^N p_{ij}\log\frac{p_{ij}}{q_{ij}+\epsilon}$$

其中，$N$是一个batch中图像或句子的数量，$p_{ij}$是**投影兼容分布**中第$i$个图像或句子与第$j$个句子或图像之间的概率，$q_{ij}$是**标准化匹配分布**中第$i$个图像或句子与第$j$个句子或图像之间的概率，$\epsilon$是一个小的正数，用于避免数值问题。

KL散度损失函数可以用来衡量两个概率分布之间的差异，当$p_{ij}$和$q_{ij}$越接近时，KL散度损失越小，表示图像和句子之间的匹配程度越高。

### Cross-Entropy Loss
Cross-Entropy Loss是一种用于衡量两个概率分布之间的差异的损失函数，它可以用于图文匹配的场景下，比如用于度量图像和文本之间的相似性或者分类。

Cross-Entropy Loss的一般形式是：

$$H(p,q)=-\sum_{x}p(x)\log q(x)$$

其中，$p$是期望输出的概率分布，$q$是实际输出的概率分布，$x$是样本空间中的一个元素。

这个损失函数的含义是，对于每个样本$x$，要使得实际输出$q(x)$接近期望输出$p(x)$，也就是说，要使得交叉熵的值越小，两个概率分布就越接近。

在Pytorch中，有一个内置的Cross-Entropy Loss函数，叫做`torch.nn.CrossEntropyLoss()`，它的定义如下：
``` python
class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)
```

它接受两个输入$input$和$target$，其中$input$是一个包含未归一化对数值（logits）的张量，形状为$(N,C)$或$(N,C,d_1,d_2,...,d_K)$，其中$N$是批次大小，$C$是类别数，$K \geq 1$是其他维度。而$target$是一个包含类别索引的张量，形状为$(N)$或$(N,d_1,d_2,...,d_K)$。它计算以下损失：

$$L=-\sum_{n=1}^Nw_{y_n}\log\frac{\exp(input_{n,y_n})}{\sum_{c=1}^C\exp(input_{n,c})}$$

其中，$w_c$是每个类别的权重（可选），如果没有指定，则默认为1。这个损失函数相当于先对$input$进行LogSoftmax操作，然后再计算负对数似然损失（NLLLoss）。它可以通过设置不同的参数来控制输出的形式，比如平均或者求和。

它的使用示例如下：
``` python
import torch
import torch.nn as nn
# 随机生成输入
input = torch.randn(3, 5)
# 设置目标类别
target = torch.tensor([1, 0, 4])
# 创建损失函数
criterion = nn.CrossEntropyLoss()
# 计算损失
loss = criterion(input, target)
print(loss)
```

输出
``` python
tensor(2.1939)
```
