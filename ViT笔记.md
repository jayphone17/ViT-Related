# ViT笔记

ViT（vision transformer）是Google在2020年提出的直接将transformer应用在图像分类的模型，后面很多的工作都是基于ViT进行改进的。ViT的思路很简单：直接把图像分成固定大小的patchs，然后通过线性变换得到patch embedding，这就类比NLP的words和word embedding，由于transformer的输入就是a sequence of token embeddings，所以将图像的patch embeddings送入transformer后就能够进行特征提取从而分类了。ViT模型原理如下图所示，其实ViT模型只是用了transformer的Encoder来提取特征（原始的transformer还有decoder部分，用于实现sequence to sequence，比如机器翻译）。

<img src="https://pic4.zhimg.com/v2-0ae5a1ed834f8007016c4492dba7e936_1440w.jpg?source=172ae18b" alt="&quot;未来&quot;的经典之作ViT：transformer is all you need!" style="zoom: 67%;" />

## **Patch Embedding**

对于ViT来说，首先要将原始的2-D图像转换成一系列1-D的patch embeddings，这就好似NLP中的word embedding。输入的2-D图像记为![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf+x%5Cin+%5Cmathbb%7BR%7D%5E%7BH%5Ctimes+W+%5Ctimes+C%7D)，其中![[公式]](https://www.zhihu.com/equation?tex=H)和![[公式]](https://www.zhihu.com/equation?tex=W)分别是图像的高和宽，而![[公式]](https://www.zhihu.com/equation?tex=C)为通道数对于RGB图像就是3。如果要将图像分成大小为![[公式]](https://www.zhihu.com/equation?tex=P%5Ctimes+P)的patchs，可以通过reshape操作得到a sequence of patchs：![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf+x_p%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes%28P%5E2%5Ccdot+C%29%7D)，图像共切分为![[公式]](https://www.zhihu.com/equation?tex=N%3DHW%2FP%5E2)个patchs，这也就是sequence的长度了，注意这里直接将patch拉平为1-D，其特征大小为![[公式]](https://www.zhihu.com/equation?tex=P%5E2%5Ccdot+C)。然后通过一个简单的线性变换将patchs映射到![[公式]](https://www.zhihu.com/equation?tex=D)大小的维度，这就是patch embeddings：![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf+%7Bx%27_%7Bp%7D%7D%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes+D%7D)，在实现上这等同于对![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf+x_p)进行一个![[公式]](https://www.zhihu.com/equation?tex=P%5Ctimes+P)且stride为![[公式]](https://www.zhihu.com/equation?tex=P)的卷积操作。

```python
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
```



## **Position Embedding**

除了patch embeddings，模型还需要另外一个特殊的position embedding。transformer和CNN不同，需要position embedding来编码tokens的位置信息，这主要是因为self-attention是permutation-invariant，即打乱sequence里的tokens的顺序并不会改变结果。如果不给模型提供patch的位置信息，那么模型就需要通过patchs的语义来学习拼图，这就额外增加了学习成本。ViT论文中对比了几种不同的position embedding方案(如下），最后发现如果不提供positional embedding效果会差，但其它各种类型的positional embedding效果都接近，这主要是因为ViT的输入是相对较大的patchs而不是pixels，所以学习位置信息相对容易很多。

- 无positional embedding
- 1-D positional embedding：把2-D的patchs看成1-D序列
- 2-D positional embedding：考虑patchs的2-D位置（x, y）
- Relative positional embeddings：patchs的相对位置

transformer原论文中是默认采用固定的positional embedding，但ViT中默认采用学习（训练的）的1-D positional embedding，在输入transformer的encoder之前直接将patch embeddings和positional embedding相加:

```python
# 这里多1是为了后面要说的class token，embed_dim即patch embed_dim
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) 

# patch emded + pos_embed
x = x + self.pos_embed
```

论文中也对学习到的positional embedding进行了可视化，发现相近的patchs的positional embedding比较相似，而且同行或同列的positional embedding也相近：

![preview](https://pic4.zhimg.com/v2-2b3cb3722c21b9df1c3f4b616f56f95f_r.jpg)

如果改变图像的输入大小，ViT不会改变patchs的大小，那么patchs的数量![[公式]](https://www.zhihu.com/equation?tex=N)会发生变化，那么之前学习的pos_embed就维度对不上了，ViT采用的方案是通过插值来解决这个问题：

```python
def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    # 除去class token的pos_embed
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    # 把pos_embed变换到2-D维度再进行插值
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb
```

但是这种情形一般会造成性能少许损失，可以通过finetune模型来解决。另外最新的论文[CPVT](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2102.10882)通过implicit Conditional Position encoding来解决这个问题（插入Conv来隐式编码位置信息，zero padding让Conv学习到绝对位置信息）。



## **Class Token**

除了patch tokens，ViT借鉴BERT还增加了一个特殊的class token。后面会说，transformer的encoder输入是a sequence patch embeddings，输出也是同样长度的a sequence patch features，但图像分类最后需要获取image feature，简单的策略是采用pooling，比如求patch features的平均来获取image feature，但是ViT并没有采用类似的pooling策略，而是直接增加一个特殊的class token，其最后输出的特征加一个linear classifier就可以实现对图像的分类（ViT的pre-training时是接一个MLP head），所以输入ViT的sequence长度是![[公式]](https://www.zhihu.com/equation?tex=N%2B1)。class token对应的embedding在训练时随机初始化，然后通过训练得到，具体实现如下：

```python
# 随机初始化
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

# Classifier head
self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

# 具体forward过程
B = x.shape[0]
x = self.patch_embed(x)
cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
x = torch.cat((cls_tokens, x), dim=1)
x = x + self.pos_embed
```



## **Transformer Encoder**

transformer最核心的操作就是self-attention，其实attention机制很早就在NLP和CV领域应用了，比如带有attention机制的seq2seq模型，但是transformer完全摒弃RNN或LSTM结构，直接采用attention机制反而取得了更好的效果：attention is all you need！简单来说，attention就是根据当前查询对输入信息赋予不同的权重来聚合信息，从操作上看就是一种“加权平均”。attention中共有3个概念：query, key和value，其中key和value是成对的，对于一个给定的query向量![[公式]](https://www.zhihu.com/equation?tex=q%5Cin+%5Cmathbb%7BR%7D%5E%7Bd%7D)，通过内积计算来匹配k个key向量（维度也是d，堆积起来即矩阵![[公式]](https://www.zhihu.com/equation?tex=K%5Cin+%5Cmathbb%7BR%7D%5E%7Bk%5Ctimes+d%7D)），得到的内积通过softmax来归一化得到k个权重，那么对于query其attention的输出就是k个key向量对应的value向量（即矩阵![[公式]](https://www.zhihu.com/equation?tex=V%5Cin+%5Cmathbb%7BR%7D%5E%7Bk%5Ctimes+d%7D)）的加权平均值。对于一系列的N个query（即矩阵![[公式]](https://www.zhihu.com/equation?tex=Q%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes+d%7D)），可以通过矩阵计算它们的attention输出：

![[公式]](https://www.zhihu.com/equation?tex=Attention%28Q%2C+K%2C+V%29+%3D+Softmax%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V+%5C%5C)

这里的![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7Bd_k%7D)为缩放因子以避免点积带来的方差影响。上述的Attention机制称为**Scaled dot product attention**，其实attention机制的变种有很多，但基本原理是相似的。如果![[公式]](https://www.zhihu.com/equation?tex=Q%2CK%2CV)都是从一个包含![[公式]](https://www.zhihu.com/equation?tex=N)个向量的sequence（![[公式]](https://www.zhihu.com/equation?tex=X%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes+D%7D)）通过线性变换得到：![[公式]](https://www.zhihu.com/equation?tex=Q%3DXW_Q%2CK%3DXW_K%2CV%3DXW_V)那么此时就变成了**self-attention**，这个时候就有![[公式]](https://www.zhihu.com/equation?tex=N)个（key,value）对，那么![[公式]](https://www.zhihu.com/equation?tex=k%3DN)。self-attention是transformer最核心部分，self-attention其实就是输入向量之间进行相互attention来学习到新特征。前面说过我们已经得到图像的patch sequence，那么送入self-attention就能到同样size的sequence输出，只不过特征改变了。

更进一步，transformer采用的是**multi-head self-attention (MSA）**，所谓的MSA就是采用定义h个attention heads，即采用h个self-attention应用在输入sequence上，在操作上可以将sequence拆分成h个size为![[公式]](https://www.zhihu.com/equation?tex=N%5Ctimes+d)的sequences，这里![[公式]](https://www.zhihu.com/equation?tex=D%3Dhd)，h个不同的heads得到的输出concat在一起然后通过线性变换得到最终的输出，size也是![[公式]](https://www.zhihu.com/equation?tex=N%5Ctimes+D)：

![[公式]](https://www.zhihu.com/equation?tex=MSA%28X%29+%3D+Concat%28head_1%2C+...%2C+head_h%29+W%5EO%2C+head_i%3DSA%28XW_i%5EQ%2C+XW_i%5EK%2C+XW_i%5EV%29+%5C%5C)

<img src="https://pic1.zhimg.com/v2-dd2b11273d3974c81d63e418bbdadbf8_r.jpg" alt="preview" style="zoom:67%;" />

在transformer中，MSA后跟一个FFN（Feed-forward network），这个FFN包含两个FC层，第一个FC层将特征从维度![[公式]](https://www.zhihu.com/equation?tex=D)变换成![[公式]](https://www.zhihu.com/equation?tex=4D)，后一个FC层将特征从维度![[公式]](https://www.zhihu.com/equation?tex=4D)恢复成![[公式]](https://www.zhihu.com/equation?tex=D)，中间的非线性激活函数采用GeLU，其实这就是一个MLP，具体实现如下：

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

那么一个完成transformer encoder block就包含一个MSA后面接一个FFN，其实MSA和FFN均包含和ResNet一样的skip connection，另外MSA和FFN后面都包含layer norm层，具体实现如下：

```python
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```



## **ViT**

对于ViT模型来说，就类似CNN那样，不断堆积transformer encoder blocks，最后提取class token对应的特征用于图像分类，论文中也给出了模型的公式表达，其中（1）就是提取图像的patch embeddings，然后和class token对应的embedding拼接在一起并加上positional embedding；（2）是MSA，而（3）是MLP，（2）和（3）共同组成了一个transformer encoder block，共有![[公式]](https://www.zhihu.com/equation?tex=L)层；（4）是对class token对应的输出做layer norm，然后就可以用来图像分类。

![img](https://pic1.zhimg.com/80/v2-cb632e9df1dbc49e379799a0417e9b34_1440w.jpg)

除了完全无卷积的ViT模型外，论文中也给出了Hybrid Architecture，简单来说就是先用CNN对图像提取特征，从CNN提取的特征图中提取patch embeddings，CNN已经将图像降采样了，所以patch size可以为![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes+1)。

ViT模型的超参数主要包括以下，这些超参数直接影响模型参数以及计算量：

1. Layers：block的数量；
2. Hidden size D：隐含层特征，D在各个block是一直不变的；
3. MLP size：一般设置为4D大小；
4. Heads：MSA中的heads数量；
5. Patch size：模型输入的patch size，ViT中共有两个设置：14x14和16x16，这个只影响计算量；

**类似BERT**，ViT共定义了3中不同大小的模型：Base，Large和Huge，其对应的模型参数不同，如下所示。如ViT-L/16指的是采用Large结构，输入的patch size为16x16。

![img](https://pic1.zhimg.com/80/v2-bf9b9ae81389a370f890b0e742de7938_1440w.jpg)



## **模型效果**

ViT并不像CNN那样具有inductive bias，论文中发现如果如果直接在ImageNet上训练，同level的ViT模型效果要差于ResNet，但是如果在比较大的数据集上petraining，然后再finetune，效果可以超越ResNet。比如ViT在Google私有的300M JFT数据集上pretrain后，在ImageNet上的最好Top-1 acc可达88.55%，这已经和ImageNet上的SOTA相当了（Noisy Student EfficientNet-L2效果为88.5%，Google最新的SOTA是Meta Pseudo Labels，效果可达90.2%）：

![img](https://pic1.zhimg.com/80/v2-c3379ed3e3fceb3776c9d8176937f738_1440w.jpg)

那么ViT至少需要多大的数据量才能和CNN旗鼓相当呢？这个论文也做了实验，结果如下图所示，从图上所示这个预训练所使用的数据量要达到100M时才能显示ViT的优势。transformer的一个特色是它的scalability：当模型和数据量提升时，性能持续提升。在大数据面前，ViT可能会发挥更大的优势。

![img](https://pic3.zhimg.com/80/v2-5486d37ee0306362fe5baa5188635656_1440w.jpg)

此外，论文中也对ViT做了进一步分析，如分析了不同layers的mean attention distance，这个类比于CNN的感受野。论文中发现前面层的“感受野”虽然差异很大，但是总体相比后面层“感受野”较小，而模型后半部分“感受野”基本覆盖全局，和CNN比较类似，说明ViT也最后学习到了类似的范式。

![img](https://pic3.zhimg.com/80/v2-28f97d96195c154e80fbb3ee76aaf8ea_1440w.jpg)

当然，ViT还可以根据attention map来可视化模型具体关注图像的哪个部分，从结果上看比较合理：

![img](https://pic1.zhimg.com/80/v2-e721a29f5231f8996c340d12027a705c_1440w.jpg)





## 图像分块嵌入

考虑到在Transformer结构中，输入是一个二维的矩阵，矩阵的形状可以表示为 ![[公式]](https://www.zhihu.com/equation?tex=%28N%2CD%29) ，其中 N 是sequence的长度，而 D 是sequence中每个向量的维度。因此，在ViT算法中，首先需要设法将 ![[公式]](https://www.zhihu.com/equation?tex=H+%2A+W+%2A+C) 的三维图像转化为 ![[公式]](https://www.zhihu.com/equation?tex=%28N%2CD%29) 的二维输入。

ViT中的具体实现方式为：将 ![[公式]](https://www.zhihu.com/equation?tex=H+%2A+W+%2A+C) 的图像，变为一个 ![[公式]](https://www.zhihu.com/equation?tex=N+%2A+%28P%5E2+%2A+C%29) 的序列。这个序列可以看作是一系列展平的图像块，也就是将图像切分成小块后，再将其展平。该序列中一共包含了 ![[公式]](https://www.zhihu.com/equation?tex=+N%3DHW%2FP%5E2+) 个图像块，每个图像块的维度则是 ![[公式]](https://www.zhihu.com/equation?tex=%28P%5E2%2AC%29) 。其中 P 是图像块的大小，C 是通道数量。经过如上变换，就可以将 N 视为sequence的长度了。

但是，此时每个图像块的维度是 ![[公式]](https://www.zhihu.com/equation?tex=%28P%5E2C%29) *，*而我们实际需要的向量维度是 D，因此我们还需要对图像块进行 Embedding。这里 Embedding 的方式非常简单，只需要对每个 ![[公式]](https://www.zhihu.com/equation?tex=%28P%5E2C%29) 的图像块做一个线性变换，将维度压缩为 D 即可。

上述对图像进行分块以及 Embedding 的具体方式如 **图** 所示。

<img src="https://pic1.zhimg.com/80/v2-f8809d6b3d1351c9fb905051b41211d0_1440w.jpg" alt="img" style="zoom:33%;" />



## 多头注意力

将图像转化为 ![[公式]](https://www.zhihu.com/equation?tex=N+%2A+%28P%5E2+%2A+C%29) 的序列后，就可以将其输入到 Transformer 结构中进行特征提取了，如 **图** 所示。

<img src="https://pic3.zhimg.com/80/v2-12c7efa557dad958946e5f684e7b1c1a_1440w.jpg" alt="img" style="zoom: 33%;" />

Transformer 结构中最重要的结构就是 Multi-head Attention，即多头注意力结构。具有2个head的 Multi-head Attention 结构如 **图** 所示。输入 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei+) 经过转移矩阵，并切分生成 ![[公式]](https://www.zhihu.com/equation?tex=q%5E%7B%28i%2C1%29%7D) 、 ![[公式]](https://www.zhihu.com/equation?tex=q%5E%7B%28i%2C2%29%7D) 、 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%7B%28i%2C1%29%7D) 、 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%7B%28i%2C2%29%7D) 、 ![[公式]](https://www.zhihu.com/equation?tex=v%5E%7B%28i%2C1%29%7D) 、 ![[公式]](https://www.zhihu.com/equation?tex=v%5E%7B%28i%2C2%29%7D) ，然后 ![[公式]](https://www.zhihu.com/equation?tex=q%5E%7B%28i%2C1%29%7D) 与 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%7B%28i%2C1%29%7D) 做 attention，得到权重向量 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ，将 ![[公式]](https://www.zhihu.com/equation?tex=+%5Calpha+%E4%B8%8E+v%5E%7B%28i%2C1%29%7D+) 进行加权求和，得到最终的 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7B%28i%2C1%29%7D%28i%3D1%2C2%2C%E2%80%A6%2CN%29) ，同理可以得到 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7B%28i%2C2%29%7D%28i%3D1%2C2%2C%E2%80%A6%2CN%29) 。接着将它们拼接起来，通过一个线性层进行处理，得到最终的结果。

<img src="https://pic3.zhimg.com/80/v2-67bbb349a67fe415e9887585d57c2f36_1440w.jpg" alt="img" style="zoom: 50%;" />

其中，使用 ![[公式]](https://www.zhihu.com/equation?tex=q%5E%7B%28i%2Cj%29%7D) 、 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%7B%28i%2Cj%29%7D) 与 ![[公式]](https://www.zhihu.com/equation?tex=v%5E%7B%28i%2Cj%29%7D) 计算 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7B%28i%2Cj%29%7D%28i%3D1%2C2%2C%E2%80%A6%2CN%29) 的方法是缩放点积注意力 (Scaled Dot-Product Attention)。 结构如 **图5** 所示。首先使用每个 ![[公式]](https://www.zhihu.com/equation?tex=q%5E%7B%28i%2Cj%29%7D) 去与 ![[公式]](https://www.zhihu.com/equation?tex=k%5E%7B%28i%2Cj%29%7D) 做 attention，这里说的 attention 就是匹配这两个向量有多接近，具体的方式就是计算向量的加权内积，得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B%28i%2Cj%29%7D) 。这里的加权内积计算方式如下所示：

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B%281%2Ci%29%7D+%3D++q%5E1+%2A+k%5Ei+%2F+%5Csqrt%7Bd%7D%5C%5C)

其中， ![[公式]](https://www.zhihu.com/equation?tex=d) 是 ![[公式]](https://www.zhihu.com/equation?tex=q) 和 ![[公式]](https://www.zhihu.com/equation?tex=k) 的维度，因为 ![[公式]](https://www.zhihu.com/equation?tex=q%2Ak) 的数值会随着维度的增大而增大，因此除以 ![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7Bd%7D) 的值也就相当于归一化的效果。

接下来，把计算得到的 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B%28i%2Cj%29%7D) 取 softmax 操作，再将其与 ![[公式]](https://www.zhihu.com/equation?tex=v%5E%7B%28i%2Cj%29%7D) 相乘。

<img src="https://pic2.zhimg.com/80/v2-cf2a49293d7ebf8f84a1ac206fe61089_1440w.jpg" alt="img" style="zoom: 67%;" />

具体代码实现如下所示。

```python
# Multi-head Attention
class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        # 计算 q,k,v 的转移矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 最终的线性层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        N, C = x.shape[1:]
        # 线性变换
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        # 分割 query key value
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Scaled Dot-Product Attention
        # Matmul + Scale
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        # SoftMax
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        # Matmul
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        # 线性变换
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```



## 多层感知机（MLP）

Transformer 结构中还有一个重要的结构就是 MLP，即多层感知机，如 **图** 所示。

<img src="https://pic3.zhimg.com/80/v2-71c83ca1915d365f81c072137f6de652_1440w.jpg" alt="img" style="zoom: 33%;" />

多层感知机由输入层、输出层和至少一层的隐藏层构成。网络中各个隐藏层中神经元可接收相邻前序隐藏层中所有神经元传递而来的信息，经过加工处理后将信息输出给相邻后续隐藏层中所有神经元。在多层感知机中，相邻层所包含的神经元之间通常使用“全连接”方式进行连接。多层感知机可以模拟复杂非线性函数功能，所模拟函数的复杂性取决于网络隐藏层数目和各层中神经元数目。多层感知机的结构如 **图** 所示。

<img src="https://pic2.zhimg.com/v2-3711949e13e7f6c2d43b3b6b2986f021_r.jpg" alt="preview" style="zoom:50%;" />

具体代码实现如下所示。

```python
class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 输入层：线性变换
        x = self.fc1(x)
        # 应用激活函数
        x = self.act(x)
        # Dropout
        x = self.drop(x)
        # 输出层：线性变换
        x = self.fc2(x)
        # Dropout
        x = self.drop(x)
        return x
```

## DropPath

除了以上重要模块意外，代码实现过程中还使用了DropPath（Stochastic Depth）来代替传统的Dropout结构，DropPath可以理解为一种特殊的 Dropout。其作用是在训练过程中随机丢弃子图层（randomly drop a subset of layers），而在预测时正常使用完整的 Graph。

具体实现如下：

```python
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
```

## 基础模块

基于上面实现的 Attention、MLP、DropPath模块就可以组合出 Vision Transformer 模型的一个基础模块，如 **图** 所示。

<img src="https://pic3.zhimg.com/80/v2-d6c7a10189f1a1532a427f1e075edcfa_1440w.jpg" alt="img" style="zoom:33%;" />

```python
class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5):
        super().__init__()
        self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        # Multi-head Self-attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        # Multi-head Self-attention， Add， LayerNorm
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Feed Forward， Add， LayerNorm
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```



## ViT网络实现

基础模块构建好后，就可以构建完整的ViT网络了。在构建完整网络结构之前，还需要介绍几个模块：

> Class Token

假设我们将原始图像切分成 ![[公式]](https://www.zhihu.com/equation?tex=3+%5Ctimes+3) 共9个小图像块，最终的输入序列长度却是10，也就是说我们这里人为的增加了一个向量进行输入，我们通常将人为增加的这个向量称为 Class Token。那么这个 Class Token 有什么作用呢？

我们可以想象，如果没有这个向量，也就是将 ![[公式]](https://www.zhihu.com/equation?tex=N%3D9) 个向量输入 Transformer 结构中进行编码，我们最终会得到9个编码向量，可对于图像分类任务而言，我们应该选择哪个输出向量进行后续分类呢？因此，ViT算法提出了一个可学习的嵌入向量 Class Token，将它与9个向量一起输入到 Transformer 结构中，输出10个编码向量，然后用这个 Class Token 进行分类预测即可。

> Positional Encoding

按照 Transformer 结构中的位置编码习惯，这个工作也使用了位置编码。不同的是，ViT 中的位置编码没有采用原版 Transformer 中的 $sincos$ 编码，而是直接设置为可学习的 Positional Encoding。对训练好的 Positional Encoding 进行可视化，如 **图9** 所示。我们可以看到，位置越接近，往往具有更相似的位置编码。此外，出现了行列结构，同一行/列中的 patch 具有相似的位置编码。

![preview](https://pic4.zhimg.com/v2-2b3cb3722c21b9df1c3f4b616f56f95f_r.jpg)

> MLP Head

得到输出后，ViT中使用了 MLP Head对输出进行分类处理，这里的 MLP Head 由 LayerNorm 和两层全连接层组成，并且采用了 GELU 激活函数。

首先构建基础模块部分，包括：参数初始化配置、独立的不进行任何操作的网络层。

```python
# 参数初始化配置
trunc_normal_ = nn.initializer.TruncatedNormal(std=.02)
zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)

# 将输入 x 由 int 类型转为 tuple 类型
def to_2tuple(x):
    return tuple([x] * 2)

# 定义一个什么操作都不进行的网络层
class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
```

完整代码如下所示。

```python
class VisionTransformer(nn.Layer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_dim=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **args):
        super().__init__()
        self.class_dim = class_dim

        self.num_features = self.embed_dim = embed_dim
        # 图片分块和降维，块大小为patch_size，最终块向量维度为768
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        # 分块数量
        num_patches = self.patch_embed.num_patches
        # 可学习的位置编码
        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)
        # 人为追加class token，并使用该向量进行分类预测
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        # transformer
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon) for i in range(depth)
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        # Classifier head
        self.head = nn.Linear(embed_dim,
                              class_dim) if class_dim > 0 else Identity()

        trunc_normal_(self.pos_emed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)
    # 参数初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        # 将图片分块，并调整每个块向量的维度
        x = self.patch_embed(x)
        # 将class token与前面的分块进行拼接
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        # 将编码向量中加入位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # 堆叠 transformer 结构
        for blk in self.blocks:
            x = blk(x)
        # LayerNorm
        x = self.norm(x)
        # 提取分类 tokens 的输出
        return x[:, 0]

    def forward(self, x):
        # 获取图像特征
        x = self.forward_features(x)
        # 图像分类
        x = self.head(x)
        return x
```

