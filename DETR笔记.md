# DETR

《End-to-End Object Detection with Adaptive Clustering Transformer》

这篇文章是第一篇使用transformer做目标检测的论文，当然它是我们前面所述的混合类型的模型．放到今天来看，DETR也存在一些缺点，尽管它在指标上可以达到FasterRCNN这样的水准，比如它在小物体检测上表现出一些能力不足的迹象，而现如今也有一些论文去改进它，比如《Deformable DETR Deformable Transformers for End-to-End Object Detection》。

![preview](https://pic1.zhimg.com/v2-2ccfe470ea6d4ae9a0d094d3385c6a24_r.jpg)

这个模型的特点是：

- 使用传统的CNN来学习2D的特征表征，同时抽取特征；
- CNN的输出被平铺出来，用来提供给transformer作为输入，这个平铺包含了特征的位置编码信息；
- transformer的输出，解码器的输出，被用来输入到一个FNN之中，然后预测类别和框．

这样的结构，相比如传统的目标检测，至少终结掉了achor的设定，并且去掉了冗余的NMS.　这些手工的操作被摒弃，尽管他们在现如今的目标检测算法中仍然发挥出巨大的作用．

DETR真正牛逼的地方，其实不是它在目标检测的效果，而是当我们把它扩展到全景分割上所展示出来的惊人效果：

![img](https://pic3.zhimg.com/80/v2-b59c8ac70808021a8241e901d816f546_1440w.jpg)

那么他们是怎么做的呢？全景分割实际上是两个任务，一个是语义分割，这个任务将类别和每个pixel对应起来，另一个是实例分割，这个任务会检测每个目标，并且将这些目标区域分割出来．而DETR将这二者结合到了一起，并且展示出了令人惊奇的效果．

在这篇论文中，一个有趣的boners是来自于这个算法对于重叠目标的区分能力，这其实也反映出了注意力机制的巨大作用，而transformer本身就是一个巨大的注意力机制引擎．比如他们可以很好的区分这些高度重叠在一起的物体：

![img](https://pic1.zhimg.com/80/v2-d3b07fab239c14309d4d5c61d2defc4c_1440w.jpg)

# Deformable DETR

1. DETR直接使用特征图进行训练，Deformable DETR使用注意力后的特征图进行训练（即每一个query搜索有效位置作为keys）
2. 重点修改了key的提取方式，以及贡献图的生成方式，贡献图直接使用query的特征回归
   Deformable DETR比DETR训练快10x
3. backbone使用resnext101-DCN-trick，在coco达到sota，尤其是小目标， APs=34.4

- **DETR存在的问题**
  1. 训练周期长，相比faster rcnn慢10-20倍
  2. 对小目标不友好。通常用多尺度特征来解小目标，然而高分辨率的特征图大大提高DETR复杂度（计算复杂度呈空间大小的平方倍上升， Nk 变多了）
- **存在上述问题的原因**
  1. 初始化时，attention model对于特征图上所有像素权重几乎是统一的（即一个query与所有的k相乘的贡献图比较均匀，也即 wq 均匀分布，然而理想结果是q与高度相关且稀疏的k相关性更强），导致需要用长训练周期学习去关注**稀疏有意义**的位置，即学习attention map前后显著的变化
  2.  处理高分辨率特征存在计算两大，存储复杂的特点。transformer中encoder的注意力权重是关于像素点的平方
- **Motivation**
  让encoder初始化的权重不再是统一分布，即不再与所有key计算相似度，而是与更有意义的key计算相似度
  **deformable convolution就是一种有效关注稀疏空间定位的方式**
  随即提出deformable detr，融合deformable conv的稀疏空间采样与transformer相关性建模能力
  在整体feature map像素中，**模型关注小序列的采样位置作为预滤波，作为key**

<img src="https://pic2.zhimg.com/v2-9fe73ed8a19328b58053f1fa73e9315e_1440w.jpg?source=172ae18b" alt="Deformable DETR 目标检测新范式！" style="zoom: 50%;" />



- **回顾DETR**
  DETR基于transformer框架，合并了set-based **匈牙利算法**，通过**二分图匹配**，强制每一个gt都有唯一的预测结果（通过该算法找优化方向，哪个gt由哪个slot负责）
  简单介绍几个概念：
  query：输出句子中的目标单词
  key：输入句子的原始单词
  cross-attention: object query从特征图（输入）中提取特征。key来自encoder的输出特征图，query来自object queries
  self-attention: object query相互影响，获取他们之间的关系。key和query都来自object queries
  Multi-head attention module: 通过度量query-key的兼容性得到的attention权重(这里权重我常称为**contribution map**)以自适应聚合关键的上下文

![img](https://pic4.zhimg.com/80/v2-9a96a0f93742b31567c8260c9421a5eb_1440w.jpg)



- **Deformable Attention Module**

对每一个query，之前关注所有的空间位置（所有位置作为key），现在只关注更有意义的、网络认为更包含局部信息的位置（少且固定数量位置作为key），缓解特征图大带来大运算量的问题
实施过程中， zq （特征图） 输入给一个线性映射，输出3MK个通道，前2MK个通道编码采样的offset，**决定每一个query应该找哪些key**，最后MK个通道，**输出keys的贡献**（不再用k×q计算，直接输入q回归），且只对找到的keys的贡献进行归一化

1. query：全图的位置 ------------------WH∗Dmodel
   key：每一个query对应的key从学习到的offset计算，文中找4个------- 4∗Dmodel
   贡献图：（即encoder中的 Softmax(KQT/√dmodel ）输入query feature后直接网络回归，相应key的位置贡献进行归一化-----------Lq∗Lk−WH∗4
   value：聚合key位置特征的query ------------ 4∗Dmodel
   output：贡献图 × value — WH∗4∗4∗Dmodel=WH∗Dmodel

![img](https://pic4.zhimg.com/80/v2-6e7ff960f3f4d2f23c31d8d9c3300b1b_1440w.jpg)



- **Multi-scale Deformable Attention Module**
  将deformable attention module扩展为多尺度feature map，主要解小目标问题
  相当于单层版本的扩展，对于一个query，每一层采集K个点作为keys，转换成，对一个query，所有层均采K个点，**融合了不同层的特征，故不需要FPN**
  这里正则化是针对一个query，所有LK个位置的贡献（回归得到）进行softmax

![img](https://pic4.zhimg.com/80/v2-a01050d69ca13df0bcf023e744ca1247_1440w.jpg)



- **Deformable Transformer Encoder**
  query为全图像素点，每个query只找4个keys（若多尺度，每个query找LK个keys），即参考点，如何取到参考点由网络学习

1. 将transformer中处理特征的部分，替换成multi-scale deformable attention module，encoder的输入输出**均为多尺度feature map**，保持相同的分辨率。
2. 从Resnet输出的C3-C5层提取多尺度特征， xlL−1l=1(L=4) ,这里**不使用FPN**， 因为每一层的query聚合了所有层key的特征
3. encoder的query不变 WHx256，key是来自多尺度特征图的像素点，即 LKx256 （每一个query的LKx256均不相同,有WH组）
4. 另外，为了定义key由哪一层级特征得到，添加了scale-level embedding，该embedding采用随机初始化，并在网络中联合训练



- **Deformable Transformer Decoder**

1. decoder由cross-attention, self-attention组成，两组attention的query相同（**要解什么就给什么，这里解的是找key的query特征，key用作预测anchor的中心**）
2. cross-attention：object queries从encoder输出的feature map中提取特征（key，value）
3. self-attention: object queries彼此交互，key与query相同
4. 针对cross-attention详细说明：
   query： 300∗dmodel 300为num_query，每一个query负责在encoder输出特征图中提取key的特征
   key： LK∗dmodel 每一个query逐L层提取K个key（输入query的feature直接回归LK个偏差，LK个贡献图）
   value： LK∗dmodel 在query基础上根据偏差聚合特征，作为value
   output：贡献图* value -> 300∗LK∗LK∗dmodel=300∗dmodel 继续串联
   每一个value都通过多尺度deformable attention module汇聚了特征，设计检测头预测的是以key为中心的bbox偏置
5. （不确定推理）文中未提及label assign问题，我估计这里与两阶段一致。根据object queries生成的keys作为anchor的中心坐标，回归偏置，生成anchor，与gt计算IOU，进行label assgin
   这样看来，若object queries有300个，即存在300×KL个anchor
   相比detr的二分图匹配，这里固定了优化的方式，加速了收敛。但提高了anchor数量



