# 读相关论文
## Bit-aware Semantic Transformer Hashing for Multi-modal Retrieval
多模态检索使用**二进制哈希码**。现有方法存在三个问题：1）浅层学习的语义表达能力有限；2）强制性的特征级多模态融合忽略了不同模态语义的差距；3）直接粗糙的成对语义保留不能有效地捕获细粒度的语义关联。

### 模型框架
模型Bit-aware Semantic Transformer Hashing (BSTH)共分为3个部分：1）Bit-aware Semantic Transformer；2）Label Prototype Learning；3）Objective Function。
![picture 1](assets/images/1685349822578.png)  

#### 符号定义
- 定义训练集为$\mathbf{O}_{tr}=\{\mathbf{o}_i\}_{i=1}^n$，训练集由$n$个图像和文本模态的样本组成。
- 利用**特定模态特征提取器**，提取出**原始图像数据和文本数据**，分别定义为$\mathbf{X}^{(I)}=[\pmb{x}_1^{(I)},\pmb{x}_2^{(I)},...,\pmb{x}_n^{(I)}] \in \mathbb{R}^{n\times d^{(I)}}$和$\mathbf{X}^{(T)}=[\pmb{x}_1^{(T)},\pmb{x}_2^{(T)},...,\pmb{x}_n^{(T)}] \in \mathbb{R}^{n\times d^{(T)}}$，其中，$d^{(I)}$和$d^{(T)}$表示图像特征和文本特征表示的**维度**。
- 定义训练集的标签矩阵为$\mathbf{L}=[\pmb{l}_1,\pmb{l}_2,...,\pmb{l}_n]\in\{0,1\}^{n\times c}$，其中$l_{ij}\in\{0,1\}$表示属于第$j$个类别的第$i$个样本的标签，$c$表示类别的数量。

#### Bit-aware Semantic Transformer
1. 首先，将**已提取的特定模态特征表示**$\pmb{x}_i^{(*)}$通过**多层感知机**$\pmb{MLP^{(*)}}(·;\theta_{mlp}^{(*)})$映射到**相同维度特征表示**$\pmb{f}_i^{(*)}$。如**公式(1)**$\pmb{f}_i^{(*)}=\pmb{MLP^{(*)}}(\pmb{x}_i^{(*)};\theta_{mlp}^{(*)}),s.t.*\in\{I,T\}$，其中$\theta_{mlp}^{(*)}$是可训练的参数。
2. 然后，定义**特征解耦层**$\pmb{DeLayer^{(*)}}(·;\theta_{de}^{(*)})$，用于分解$\pmb{f}_i^{(*)}\in\mathbb{R}^{1\times d_c}$为序列$\mathbf{C}_i^{(*)}$，序列中包含$k$个**粗隐式语义概念表示**。如**公式(2)**$\mathbf{C}_i^{(*)}=[\pmb{c}_{i1}^{(*)},\pmb{c}_{i2}^{(*)},...,\pmb{c}_{ik}^{(*)}]=\pmb{DeLayer^{(*)}}(\pmb{f}_i^{(*)};\theta_{de}^{(*)})$，其中$\pmb{c}_{ik}^{(*)}\in\mathbb{R}^{1\times d_c}$为第$i$个**特定模态样本**的第$k$个**粗隐式语义概念表示**，$d_c$表示某**概念表示**的**维度**，$\theta_{de}^{(*)}$为可训练参数。
3. 进一步，引入`Modality-specific Transformer Encoder`，用于以`self-attention`方式捕获这$k$个**粗粒度隐式语义概念**之间的**潜在相关性**。如**公式(3)**$\mathbf{\widetilde{C}}_i^{(*)}=\pmb{TransformerEncoder^{(*)}}(\mathbf{C}_i^{(*)};\theta_{enc}^{(*)})$，其中，$\mathbf{\widetilde{C}}_i^{(*)}\in\mathbb{R}^{k\times d_c}$定义为第$i$个样本的**细粒度的语义概念序列**，$\theta_{enc}^{(*)}$为可训练的参数。
4. 然后，进行多模态融合。如**公式(4)**$\mathbf{\widetilde{C}}_i^{f}=[\pmb{\widetilde{c}}_{i1}^{f},\pmb{\widetilde{c}}_{i2}^{f},...,\pmb{\widetilde{c}}_{ik}^{f}]=\sum\limits_{*\in\{I,T\}}\mathbf{\widetilde{C}}_i^{(*)},\mathbf{\widetilde{C}}_i^{f}\in\mathbb{R}^{k\times d_c}$，其中，$\mathbf{\widetilde{C}}_i^{f}$为第$i$个样本的**已融合的隐式语义概念序列**。
5. 最后，采用`bit-wise hash functions`对已融合的隐式语义概念表示编码为**相应的哈希位**。如**公式(6)**$\pmb{h}_i=\pmb{Concat}(\mathcal{H}_1(\pmb{\widetilde{c}}_{i1}^{f};\theta_{h1}),\mathcal{H}_2(\pmb{\widetilde{c}}_{i2}^{f};\theta_{h2}),...,\mathcal{H}_k(\pmb{\widetilde{c}}_{ik}^{f};\theta_{hk}))$和**公式(7)**$\pmb{b}_i=sign(\pmb{h}_i),\pmb{h}_i\in\mathbb{R}^{1\times k},\pmb{b}_i\in\{-1,1\}^{1\times k}$，其中，$\theta_{hk}$为可训练的参数，$\mathcal{H}_1(·;\theta_{h1})$为与第$k$个哈希位相关的`bit-wise hash function`。
6. 另外，为了保留语义信息（没看懂）。如**公式(8)**$\widetilde{\pmb{l}}_i=sigmoid(\pmb{FC}(\pmb{h}_i;\theta_{fc}))$，其中，$\pmb{FC}(·;\theta_{fc})$为**全连接层**，$\theta_{fc}$为可训练的参数。

#### Label Prototype Learning
1. 首先，将所有类别投影到`feature embeddings`。如**公式(9)**$\mathbf{E}^{\pmb{L}}=\pmb{Embedding}(\mathcal{S}^{\pmb{L}};\theta_{el}),\mathbf{E}^{\pmb{L}}\in\mathbb{R}^{c\times k}$，其中，$\mathcal{S}^{\pmb{L}}$为类别序列（例如["dog","cat","person",...]），$\theta_{el}$为可训练参数。
2. 然后，为了学习`label prototype embeddings`，保留**不同类别**间的**显式语义关联**，引入`transformer encoder`。如**公式(10,11)**$\mathbf{\widetilde{E}}^{\pmb{L}}=[\pmb{\widetilde{e}}_{1}^{\pmb{L}},\pmb{\widetilde{e}}_{2}^{\pmb{L}},...,\pmb{\widetilde{e}}_{c}^{\pmb{L}}]=\pmb{TransformerEncoder^{(\pmb{L})}}(\mathbf{E}^{\pmb{L}};\theta_{enc}^{\pmb{L}}),\mathbf{\widetilde{E}}^{\pmb{L}}\in\mathbb{R}^{c\times k}$，其中，$\theta_{enc}^{\pmb{L}}$为可训练参数。
3. 进一步，通过`category-wise hash function`学习最终的`label prototype embeddings`。如**公式(12)**$\mathbf{P}^{\pmb{L}}=[\pmb{p}_1^{\pmb{L}},\pmb{p}_2^{\pmb{L}},...,\pmb{p}_c^{\pmb{L}}]=[\mathcal{H}_1^p(\pmb{\widetilde{e}}_{1}^{\pmb{L}};\theta_{h1}^p),\mathcal{H}_2^p(\pmb{\widetilde{e}}_{2}^{\pmb{L}};\theta_{h2}^p),...,\mathcal{H}_c^p(\pmb{\widetilde{e}}_{c}^{\pmb{L}};\theta_{hc}^p)]$，其中，$\mathbf{P}^{\pmb{L}}\in\mathbb{R}^{c\times k}$定义为所有的`label prototype embeddings`，$\theta_{hc}^p$为可训练参数。
4. 然后，为了防止`sign(·)`操作带来的量化误差，采用了`relaxed label prototype embeddings`。在学习了`label prototype embeddings`$\mathbf{P}^{L}$之后，使用标签向量$\pmb{l}_i$通过与$\mathbf{P}^{L}$线性组合生成**监督哈希码**。如**公式(13)**$\pmb{h}_i^L=\pmb{l}_i\mathbf{P}^L,\pmb{h}_i^L\in\mathbb{R}^{1\times k}$和**公式(14)**$\pmb{b}_i^L=sign(\pmb{h}_i^L),\pmb{b}_i^L\in\{-1,1\}^{1\times k}$。
5. 最后，类似**公式(8)**预测**伪标签**。如**公式(15)**$\widetilde{\pmb{l}}_i^L=sigmoid(\pmb{FC}^{(L)}(\pmb{h}_i^L;\theta_{fc}^L))$。

#### 目标函数
对两个过程（`Bit-aware Semantic Transformer`和`Label Prototype Learning`）分别定义了不同的**损失函数**。

##### Bit-aware Semantic Transformer
分类损失$\mathcal{L}_{clf}$，用于将已标注语义信息保存到**已预测的伪标签**中。如**公式(16)**$\mathcal{L}_{clf}=||\widetilde{\pmb{l}}_i-\pmb{l}_i||_2^2$。

符号损失$\mathcal{L}_{sign}$，用于最小化`sign(·)`操作的量化误差。如**公式(17)**$\mathcal{L}_{sign}=||\pmb{h}_i-\pmb{b}_i^L||_2^2$。

相似度损失$\mathcal{L}_{sim}$，用于保持样本间的两两相关性。如**公式(18)**$\mathcal{L}_{sim}=||cos(\pmb{h}_i,\pmb{h}_j)-\mathbf{S}_{ij}||_2^2$，其中$\mathbf{S}$是相似度矩阵，其建模了相关样本之间细粒度的关联，可以表示为**公式(19)**$\mathbf{S}_{ij}=\frac{2}{1+e^{-\pmb{l}_i\pmb{l}_j^T}}-1,\mathbf{S}_{ij}\in[0,\frac{2}{1+e^{-c}}-1]$。

最后，得到此学习阶段(`Bit-aware Semantic Transformer`)的目标函数为**公式(20)**$\min\limits_{\Theta_{BaT}}\mathcal{L}=\beta_1\mathcal{L}_{clf}+\beta_2\mathcal{L}_{sign}+\beta_3\mathcal{L}_{sim}$，其中，$\beta_1,\beta_2,\beta_3$是`trade-off hyper-parameters`，$\Theta_{BaT}$表示该模块(`Bit-aware Semantic Transformer`)中所有可训练的参数。


