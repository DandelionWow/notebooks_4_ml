# 读相关论文
## Bit-aware Semantic Transformer Hashing for Multi-modal Retrieval
多模态检索使用**二进制哈希码**。现有方法存在三个问题：1）浅层学习的语义表达能力有限；2）强制性的特征级多模态融合忽略了不同模态语义的差距；3）直接粗糙的成对语义保留不能有效地捕获细粒度的语义关联。

### 模型框架
模型Bit-aware Semantic Transformer Hashing (BSTH)共分为3个部分：1）Bit-aware Semantic Transformer；2）Label Prototype Learning；3）Objective Function。
![picture 1](assets/images/1685349822578.png)  

#### 符号定义
- 定义训练集为$\mathbf{O}_{tr}=\{\mathbf{o}_i\}_{i=1}^n$，训练集由$n$个图像和文本模态的样本组成。
- 利用**特定模态特征提取器**，提取出**原始图像数据和文本数据**，分别定义为$\mathbf{X}^{(I)}=[\pmb{x}_1^{(I)},\pmb{x}_2^{(I)},...,\pmb{x}_n^{(I)}] \in \mathbb{R}^{n\times d^{(I)}}$和$\mathbf{X}^{(T)}=[\pmb{x}_1^{(T)},\pmb{x}_2^{(T)},...,\pmb{x}_n^{(T)}] \in \mathbb{R}^{n\times d^{(T)}}$，其中，$d^{(I)}$和$d^{(T)}$表示图像特征和文本特征表示的**维度**。

#### Bit-aware Semantic Transformer
##### 第一步，feature decoupling layer
1. 首先，将**已提取的特定模态特征表示**$\pmb{x}_i^{(*)}$通过**多层感知机**$\pmb{MLP^{(*)}}(·;\theta_{mlp}^{(*)})$映射到**相同维度特征表示**$\pmb{f}_i^{(*)}$。如**公式(1)**$\pmb{f}_i^{(*)}=\pmb{MLP^{(*)}}(\pmb{x}_i^{(*)};\theta_{mlp}^{(*)}),s.t.*\in\{I,T\}$，其中$\theta_{mlp}^{(*)}$是可训练的参数。
2. 然后，定义**特征解耦层**$\pmb{DeLayer^{(*)}}(·;\theta_{de}^{(*)})$，用于分解$\pmb{f}_i^{(*)}\in\mathbb{R}^{1\times d_c}$为序列$\mathbf{C}_i^{(*)}$，序列中包含$k$个**粗隐式语义概念表示**。如**公式(2)**$\mathbf{C}_i^{(*)}=[\pmb{c}_{i1}^{(*)},\pmb{c}_{i2}^{(*)},...,\pmb{c}_{ik}^{(*)}]=\pmb{DeLayer^{(*)}}(\pmb{f}_i^{(*)};\theta_{de}^{(*)})$，其中$\pmb{c}_{ik}^{(*)}\in\mathbb{R}^{1\times d_c}$为第$i$个**特定模态样本**的第$k$个**粗隐式语义概念表示**，$d_c$表示某**概念表示**的**维度**，$\theta_{de}^{(*)}$为可训练参数。
3. 进一步，引入`Modality-specific Transformer Encoder`，用于以`self-attention`方式捕获这$k$个**粗粒度隐式语义概念**之间的**潜在相关性**。如**公式(3)**$\mathbf{\widetilde{C}}_i^{(*)}=\pmb{TransformerEncoder^{(*)}}(\mathbf{C}_i^{(*)};\theta_{enc}^{(*)})$，其中，$\mathbf{\widetilde{C}}_i^{(*)}\in\mathbb{R}^{k\times d_c}$定义为第$i$个样本的**细粒度的语义概念序列**，$\theta_{enc}^{(*)}$为可训练的参数。
4. 然后，进行多模态融合。如**公式(4)**$\mathbf{\widetilde{C}}_i^{f}=[\pmb{\widetilde{c}}_{i1}^{f},\pmb{\widetilde{c}}_{i2}^{f},...,\pmb{\widetilde{c}}_{ik}^{f}]=\sum\limits_{*\in\{I,T\}}\mathbf{\widetilde{C}}_i^{(*)},\mathbf{\widetilde{C}}_i^{f}\in\mathbb{R}^{k\times d_c}$，其中，$\mathbf{\widetilde{C}}_i^{f}$为第$i$个样本的**已融合的隐式语义概念序列**。
5. 最后，采用`bit-wise hash functions`对已融合的隐式语义概念表示编码为**相应的哈希位**。如**公式(6)**$\pmb{h}_i=\pmb{Concat}(\mathcal{H}_1(\pmb{\widetilde{c}}_{i1}^{f};\theta_{h1}),\mathcal{H}_2(\pmb{\widetilde{c}}_{i2}^{f};\theta_{h2}),...,\mathcal{H}_k(\pmb{\widetilde{c}}_{ik}^{f};\theta_{hk}))$和**公式(7)**$\pmb{b}_i=sign(\pmb{h}_i),\pmb{h}_i\in\mathbb{R}^{1\times k},\pmb{b}_i\in\{-1,1\}^{1\times k}$，其中，$\theta_{hk}$为可训练的参数，$\mathcal{H}_1(·;\theta_{h1})$为与第$k$个哈希位相关的`bit-wise hash function`。
6. 另外，为了保留语义信息（没看懂）。如**公式(8)**$\widetilde{\pmb{l}}_i=sigmoid(\pmb{FC}(\pmb{h}_i;\theta_{fc}))$，其中，$\pmb{FC}(·;\theta_{fc})$为**全连接层**，$\theta_{fc}$为可训练的参数。