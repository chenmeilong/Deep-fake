# Deep-fake
## 应用
DeepFake是指使用深度学习技术生成逼真的伪造视频、音频或图像，从而让人产生一种虚假的印象。尽管DeepFake技术有一些负面的应用，但也有一些潜在的正面应用。以下是一些DeepFake技术的应用领域：
1. 娱乐业：DeepFake可以用于电影、电视剧或广告中的特效制作，使角色看起来更加逼真或实现特殊效果。
2. 影视修复：DeepFake可以用于修复旧电影或照片中的损坏或缺失的部分，使其看起来更加完整。
3. 视频游戏：DeepFake可以用于游戏中的角色动画，使其动作和表情更加逼真。
教育和研究：DeepFake可以用于教育和研究领域，例如创建虚拟人物来模拟不同情境下的交互或研究人类行为。
4. 艺术创作：DeepFake可以用于创意艺术作品的生成，例如生成逼真的数字艺术或虚拟角色。
![](http://img.cmlt.fun/article/20230616122805.png)

## 原理
换脸原理：**用监督学习训练一个神经网络将张三的扭曲处理过的脸还原成原始脸，并且期望这个网络具备将任意人脸还原成张三的脸的能力。**

![](http://img.cmlt.fun/article/20230616123130.png)
![](http://img.cmlt.fun/article/20230616123208.png)

### 自编码机（AutoEncoder)
自编码器是一种能够通过无监督学习，学到输入数据高效表示的人工神经网络。输入数据的这一高效表示称为编码，其维度一般远小于输入数据，使得自编码器可用于降维。更重要的是，自编码器可作为强大的特征检测器（feature detectors），应用于深度神经网络的预训练。此外，自编码器还可以随机生成与训练数据类似的数据，这被称作生成模型（generative model）。

Deepfake实现流程：一提取数据，二训练，三转换。其中第一和第三步都需要用到数据预处理，另外第三步还用到了图片融合技术。


## 代码解析

### 目录结构
![](http://img.cmlt.fun/article/20230616123758.png)

### 图像预处理
从大图中识别，并抠出人脸图像，根据这些坐标能计算人脸的角度，最终抠出来的人脸是摆正后的人脸。

```python
def random_warp(image): #从对齐的人脸图像中得到一对随机变形的图像  
	assert image.shape == (256, 256, 3)
	range_ = numpy.linspace(128 - 80, 128 + 80, 5)
	mapx = numpy.broadcast_to(range_, (5, 5))
	mapy = mapx.T
	mapx = mapx + numpy.random.normal(size=(5, 5), scale=5)
	mapy = mapy + numpy.random.normal(size=(5, 5), scale=5)
	interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
	interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

	warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)     
    #重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
	src_points = numpy.stack([mapx.ravel(), mapy.ravel()], axis=-1)
	dst_points = numpy.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
	mat = umeyama(src_points, dst_points, True)[0:2]       #得到  仿射变换
	target_image = cv2.warpAffine(image, mat, (64, 64))#仿射变换    image 输入图像。 mat 变换矩阵
	return warped_image, target_image
```

### 模型结构

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            _ConvLayer(3, 128),
            _ConvLayer(128, 256),
            _ConvLayer(256, 512),
            _ConvLayer(512, 1024),
            Flatten(),
            nn.Linear(1024 * 4 * 4, 1024),
            nn.Linear(1024, 1024 * 4 * 4),
            Reshape(),
            _UpScale(1024, 512),
        )

        self.decoder_A = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

        self.decoder_B = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, select='A'):
        if select == 'A':
            out = self.encoder(x)    #编码      torch.Size([32, 512, 8, 8])
            out = self.decoder_A(out)  #解码         类似反卷积还原成原来的图像  但是不是反卷积算法
        else:
            out = self.encoder(x)    #编码      torch.Size([32, 512, 8, 8])
            out = self.decoder_B(out)   #解码         类似反卷积还原成原来的图像  但是不是反卷积算法
        return out
```

### 图像融合
成功训练之后，自动编码机会将截取下来的明星A的脸部细节换成明星B的脸部，同时保留A的眼睛，鼻子，嘴巴的位置以及表情，最后一步是需要将换好的脸部截图贴回原图，并与原图天衣无缝的融合。
    分别计算两个明星头像的 R, G, B 三颜色通道所有像素的平均值，将背景的 R，G，B 通道的像素均值做平移，来达到与脸部截图匹配的效果，这种方法称作 Average Color Adjust，简单粗暴。

```python
images_A = get_image_paths("data/trump") #图片路径列表
images_B = get_image_paths("data/cage")
images_A = load_images(images_A) / 255.0 #图片列表 (376,256,256,3) #trump 
images_B = load_images(images_B) / 255.0 #(318, 256, 256, 3)     #cage
# cv2.imshow("img", images_A[0])
images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2)) 
#简单粗暴  求三通道均值  将A图的色调变为B的色调  其他不变
```


## 效果
![](http://img.cmlt.fun/article/20230616132625.png)


## 总结
源码[点这](https://github.com/chenmeilong/Deep-fake)，autoencoder.t7文件[点这](http://img.cmlt.fun/article/autoencoder.t7),本本环境使用的pytorch1.1，其他版应该也是兼容的。
 单纯从技术的层面上来看，Deepfake是一个很不错的应用，在电影制作，录制回忆片，纪录片中发挥作用，真实地还原历史人物的原貌，这可能是无法仅由演员和化妆师做到的。

***
**如果你觉得本文对你有所帮助，别忘记给我点个start，有任何疑问和想法，欢迎在评论区与我交流。**
