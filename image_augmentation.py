import cv2
import numpy

from umeyama import umeyama

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):   #随机数据增强   图片扭曲变形
    h, w = image.shape[0:2]
    rotation = numpy.random.uniform(-rotation_range, rotation_range)     #取随机数
    scale = numpy.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = numpy.random.uniform(-shift_range, shift_range) * w
    ty = numpy.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)        #中心旋转，旋转角度 ，缩放比例
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if numpy.random.random() < random_flip:
        result = result[:, ::-1]
    return result


# get pair of random warped images from aligened face image  从对齐的人脸图像中得到一对随机变形的图像
def random_warp(image):
    assert image.shape == (256, 256, 3)
    range_ = numpy.linspace(128 - 80, 128 + 80, 5)
    mapx = numpy.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    mapx = mapx + numpy.random.normal(size=(5, 5), scale=5)
    mapy = mapy + numpy.random.normal(size=(5, 5), scale=5)

    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)     #重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。

    src_points = numpy.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = numpy.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]       #得到  仿射变换

    target_image = cv2.warpAffine(image, mat, (64, 64))         #仿射变换    image 输入图像。 mat 变换矩阵。

    return warped_image, target_image
