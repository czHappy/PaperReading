# AR/平面检测路线调研

## AR SDK调用

### 基本思路

- 基于二维码和二维图片的识别跟踪技术已经基本成熟，也有了广泛的应用，算法改进的主要目标还是在于提高算法的稳定性和准确性。
- 首先进行模型注册，模型注册的主要目标是建立三维物体的特征点的三维坐标信息库。
- 在检测过程中，拍摄视频画面，检测图像的自然特征，将当前视频图像与指定模型的参考图像匹配，根据匹配结果，判断当前场景图像与模型图像是否相同。
- 如果不相同，则继续识别过程。否则，进入到检测阶段。
- 在检测阶段，根据映射表找到当前图像对应的物体模型的 3D 点坐标，得到 2D 坐标到 3D 空间坐标的投影矩阵，根据投影矩阵和已知的内参矩阵恢复出当前图像帧的位姿矩阵，然后定义虚拟物体的坐标系，叠加三维虚拟物体进行渲染。
- 此后开始进入到跟踪阶段进行跟踪计算新的位姿矩阵，当跟踪到的点数影响到了计算位姿矩阵的精度时，则重新进行识别和检测。

### ARCore + Unity

- 实现特定物体/平面上的虚拟物体描绘 https://www.bilibili.com/video/BV1XJ411R7oA/?spm_id_from=333.788.recommend_more_video.5
- 各种 AR效果：https://mp.weixin.qq.com/s/M9qPpriURNjcC99MeYIcng
- 运动追踪：ARCore 的运动跟踪技术使用手机摄像头标识兴趣点（称为特征点），并跟踪这些点随着时间变化的移动。将这些点的移动数据与手机惯性传感器的读数组合，即使手机一直移动，也能确定位置和屏幕方向。
- 理解融合：除了标识关键点外，ARCore 还会检测平坦的表面（例如桌子或地面），并估测周围区域的平均光照强度。 这些功能共同构成了 ARCore 对现实世界的理解。

### Vuforia SDK + Unity

- 实现特定物体/平面上的虚拟物体描绘
  - https://www.bilibili.com/video/BV1AJ411i7xP?p=1

### Wikitude Markerless AR SLAM + unity

- 虚拟物体放置
  - https://www.bilibili.com/video/BV1s4411N74U?from=search&seid=4624192271448881857

### EasyAR Sense

- https://www.easyar.cn/view/sdk.html




## 平面检测算法调研

### PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image

- [CVPR 2019](https://paperswithcode.com/conference/cvpr-2019-6)

- 论文地址：https://arxiv.org/pdf/1812.04072.pdf
- 代码地址：https://github.com/art-programmer/PlaneNet
- 主要贡献：Accurate 3D Ground Plane Estimation from a Single Image提出了一种深度神经网络结构 PlaneRCNN，它可以从单个RGB图像中检测和重建分段平面。PlaneRCNN 为了检测出平面的平面参数和分割掩膜而采用了 Mask R-CNN 的变种算法。然后，PlaneRCNN 联合细化所有的分割掩膜，在训练期间形成一个新的 loss，强制得与该 loss 就近的视图保持一致。本文还提出了一个新的基准用于在真实样本中能有更细粒度的平面分割；

### Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding

- 论文地址：https://arxiv.org/pdf/1902.09777.pdf

- 代码：https://github.com/svip-lab/PlanarReconstruction

- 主要贡献：单图像分段平面3D重建旨在同时分割平面实例并从图像恢复3D平面参数。本文提出了一种基于关联嵌入的新颖的两阶段方法，在第一阶段，训练CNN将每个像素映射到嵌入空间，其中来自同一平面实例的像素具有类似的嵌入。然后，通过有效的均值漂移聚类算法将嵌入矢量分组到平面区域中来获得平面实例。在第二阶段，通过考虑像素级和实例级一致性来估计每个平面实例的参数。通过所提出的方法能够检测任意数量的平面。

  

### Accurate 3D Ground Plane Estimation from a Single Image

- 论文地址：https://lear.inrialpes.fr/people/cherian/papers/3dpaperICRA09.pdf
- 代码地址：无

- 主要贡献：首先从单个图像准确地重建3D环境，然后在环境上定义坐标系统，然后再使用环境特征针对此坐标系执行所需的定位。从给定图像中准确估计出地平面，然后将图像分割成地面和垂直区域。执行基于马尔可夫随机场（MRF）的3D重建，以构建给定图像的近似深度图。

## 其他

### **SuperPlane: 3D Plane Detection and Description from a Single Image**

- 2021 IEEE Virtual Reality and 3D User Interfaces (VR)
- 论文地址：https://conferences.computer.org/vrpub/pdfs/VR2021-2AyvgnPUHcYon9QQHz6BPD/255600a207/255600a207.pdf

### Joint tracking and ground plane estimation



## SLAM 框架

### ORB SLAM系列

- 包括视觉里程计、跟踪、回环检测，是一种完全基于稀疏特征点的单目 SLAM 系统，同时还有单目、双目、RGBD 相机的接口。其核心是使用 ORB (Orinted FAST and BRIEF) 作为整个视觉 SLAM 中的核心特征。
  - 跟踪（Tracking） ： 这一部分主要工作是从图像中提取 ORB 特征，根据上一帧进行姿态估计，或者进行通过全局重定位初始化位姿，然后跟踪已经重建的局部地图，优化位姿，再根据一些规则确定新关键帧。
  -  建图（LocalMapping） ：这一部分主要完成局部地图构建。包括对关键帧的插入，验证最近生成的地图点并进行筛选，然后生成新的地图点，使用局部捆集调整（Local BA），最后再对插入的关键帧进行筛选，去除多余的关键帧。
  - 闭环检测（LoopClosing） ：这一部分主要分为两个过程，分别是闭环探测和闭环校正。闭环检测先使用 WOB 进行探测，然后通过 Sim3 算法计算相似变换。闭环校正，主要是闭环融合和 Essential Graph 的图优化。
- 相关资料：
- markerless ar 结合 orb-slam http://paopaorobot.org/794.html 虚拟物体创建，把markless出来的虚拟人物加到slam坐标系

![](https://pic2.zhimg.com/80/9e3c8d2d5fd53a6d66f0b94b86818cfd_1440w.png)

### AR 论文

###  HDR Environment Map Estimation for Real-Time Augmented Reality

![截屏2021-08-27 上午9.46.16](/Users/cz/Library/Application Support/typora-user-images/截屏2021-08-27 上午9.46.16.png)

### FAST DEPTH DENSIFICATION FOR OCCLUSION-AWARE AUGMENTED REALITY

- 2018 ACM Facebook
- https://homes.cs.washington.edu/~holynski/publications/occlusion/index.html
- code: https://github.com/facebookresearch/AR-Depth

![截屏2021-08-27 上午9.57.20](/Users/cz/Library/Application Support/typora-user-images/截屏2021-08-27 上午9.57.20.png)



![截屏2021-08-27 上午9.57.45](/Users/cz/Library/Application Support/typora-user-images/截屏2021-08-27 上午9.57.45.png)



### Consistent Video Depth Estimation

- 主页/视频/源码： https://roxanneluo.github.io/Consistent-Video-Depth-Estimation/

![截屏2021-08-27 上午10.07.13](/Users/cz/Library/Application Support/typora-user-images/截屏2021-08-27 上午10.07.13.png)

### 总结

- 基于深度学习的AR主要是估计每个像素点的距离 We need to know how far away every pixel is.

![截屏2021-08-27 上午9.54.08](/Users/cz/Library/Application Support/typora-user-images/截屏2021-08-27 上午9.54.08.png)

![截屏2021-08-27 上午9.54.23](/Users/cz/Library/Application Support/typora-user-images/截屏2021-08-27 上午9.54.23.png)

