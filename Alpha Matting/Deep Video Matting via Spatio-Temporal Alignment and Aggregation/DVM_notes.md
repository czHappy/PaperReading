# Deep Video Matting via Spatio-Temporal Alignment and Aggregation 

## 创新点

- ST-FAM is designed to effectively align and aggregate information across different spatial scales and temporal frames within the network decoder* 

- *a large-scale video matting dataset with groundtruth alpha mattes for quantitative evaluation and real-world high- resolution videos with trimaps for qualitative evaluation.* 

- a novel correlation layer is introduced to propagate trimaps across different frames

* sub-pixel convolution layer to upsample features, rather than unpooling operation or deconvolution, for both accuracy and efficiency: unpooling operation generates sparse indices and sometimes leads to zero gradients, while deconvolution suffers efficiency problem.* 
* *deformable convolution.* for feature alignment.



- *https://github.com/nowsyn/DVM* *.* 

## 视频抠图的两个挑战点

- First, video matting needs to preserve spa- tial and temporal coherence in the predicted alpha matte. 

- 单帧抠图不可避免地形成闪烁
  - 实验了光流，即使是最好的光流方法都不行，原因是no good optical flow estima- tion that can handle large area of semi-transparency. 
  - The other challenge for video matting is the necessary input of a dense trimap for each frame 



## 数据集方面

- 使用了合成数据集,前景是找的绿幕，用软件很容易抠出来，背景是一些自然场景。
  - For foreground video objects, we collect available green screen video clips from the Internet, from which we extract foreground color and alpha matte using a chroma keying software provided by Foundry Keylight
  - The background set consists of various real-life videos. We collect over 6500 free video clips of natural scenarios, city views and indoor environment from the Internet. 



## 方法

- trimap propagation 给定目标帧和参考帧，通过关联层计算像素间的相似度，输出融合特征图，再经过解码器得到当前帧的trimap；
- 训练时间隔前后n帧图像和参考帧同时做trimap propagation 得到trimap，经过类U-Net网络（连接层使用ST- FAM）最终得到alpha
- 使用spatio-temporal feature aggregation module (ST-FAM) 利用时序信息，由temporal feature alignment module (TFA) and temporal feature fusion module (TFF). 组成。

1. - 视频抠图的一大优势是像素的运动信息可以指导分辨前后景。Temporal Feature Alignment Module 通过对像素点p的附近帧的一些偏移位置进行卷积（deformable convolution ）
   - Temporal Feature Fusion Module 实际上就是通道注意力机制和空间注意力，将对齐后的特征图进行一个空间注意力的强化，形成最终的特征图输出



![截屏2021-11-09 下午4.11.36](/Users/cz/Documents/论文阅读/去背景方案总结/img/截屏2021-11-09 下午4.11.36.png)

![截屏2021-11-09 下午4.12.00](/Users/cz/Documents/论文阅读/去背景方案总结/img/截屏2021-11-09 下午4.12.00.png)

## 损失函数

- Composition Loss. 就是组合后的图片和ground truth之间的L1 差距

- Sobel filter 计算alpha_t的图像梯度差距

-  KL散度 计算alpha_t和ground truth的KL散度差距（归一化之后）

- 时序一致性损失alpha_t对t的导数差异

![截屏2021-11-09 下午4.12.32](/Users/cz/Documents/论文阅读/去背景方案总结/img/截屏2021-11-09 下午4.12.32.png)

