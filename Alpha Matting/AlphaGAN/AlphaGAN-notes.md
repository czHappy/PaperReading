# **AlphaGAN: Generative adversarial networks for natural image matting**

## Article

- Lutz S, Amplianitis K, Smolic A. Alphagan: Generative adversarial networks for natural image matting[J]. arXiv preprint arXiv:1807.10088, 2018.

## Main Ideas

- Over-dependency solely on color information can lead to artifacts in images where the foreground and background color distributions overlap.

- Propose a generative adversarial network (GAN) for natural image matting. 
- Improve on the network architecture of Xu et al (Deep image matting.  CVPR 2017). to better deal with the spatial localization issues inherent in CNNs by using dilated convolutions to capture global context information without downscaling feature maps and losing spatial information.
  - Use the decoder structure of the network as the generator in our generative adversarial model. 
  - The discriminator is trained on images that have been composited with the ground-truth alpha and the predicted alpha.
  - The discriminator learns to recognize images that have been composited well, which helps the generator learn alpha predictions that lead to visually appealing compositions.
-  This work is the first approach using generative adversarial neural networks for natural image matting.
  -  GANs have shown good performance in other computer vision tasks, such as image-to-image translation,  image generation or image editing.

## Method

### Dataset

- DIM dataset contains 431 unique foreground objects and their corresponding alpha.
- Random background image is selected from MSCOCO.
- Generate a trimap by dilating the ground-truth alpha.
- Random crop and rotation are used for data augmentation.



### Generator

- The generator consists of a an encoder-decoder network similar to those that have achieved good results in other computer vision tasks, such as semantic segmentation.
  - dilated convolutions
    - Adding pooling layer leads to information loss
    - No pooling layers leads to receptive field decreasing which causes CNN difficult to learn global feature .
    - No pooling layers and expanding conv kernels leads to huge computation cost.
    - dilated convolutions  enlarges receptive field while keeping small computation.
  - ASPP(atrous spatial pyramid pooling)
    - Caputure context information of image by multiple scales.
  - Skip connection
    - Make deep CNN training easier.
- Input
  - *G* takes an image composited from the foreground, alpha and a random background appended with the trimap as 4th-channel as input and attempts to predict the correct alpha. 

![截屏2021-11-11 下午3.20.22](img/截屏2021-11-11 下午3.20.22.png)

### Discriminator

-  Use the PatchGAN.
  - This discriminator attempts to classify every *N* × *N* patch of the input as real or fake. The discriminator is run convolutionally over the input and all responses are averaged to calculate the final prediction of the discriminator *D*.
- The input of *D* consists of 4 channels. 
  - The first 3-channels consist of the RGB values of a newly composited image, using the ground-truth foreground, a random background and the predicted alpha. 
  - The 4th channel is the input trimap to help guide the discriminator to focus on salient regions in the image.

### Loss Function

- Alpha Loss

- Composition Loss

- GAN Loss
  $$
  \mathcal{L}_{G A N}(G, D)=\log D(x)+\log (1-D(C(G(x)))
  $$

- Total Loss

$$
\mathcal{L}_{\text {AlphaGAN }}(G, D)=\mathcal{L}_{\text {alpha }}(G)+\mathcal{L}_{\text {comp }}(G)+\mathcal{L}_{\text {GAN }}(G, D)
$$

- Object

$$
\arg \min _{G} \max _{D} \mathcal{L}_{A l p h a G A N}
$$

## Discuss

- Use GAN as auxiliary for other tasks.
  - Segmentation
  - Detection
  - Image Composition

