import torch
import torch.utils.data
from torch import nn, optim
from padding_same_conv import Conv2d


def toTensor(img):
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img


def var_to_np(img_var):
    return img_var.data.cpu().numpy()


class _ConvLayer(nn.Sequential):
    def __init__(self, input_features, output_features):
        super(_ConvLayer, self).__init__()
        self.add_module('conv2', Conv2d(input_features, output_features,
                                        kernel_size=5, stride=2))
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))


class _UpScale(nn.Sequential):  # 类似反卷积操作   但是不能用反卷积代替   这种算法非常巧妙
    def __init__(self, input_features, output_features):
        super(_UpScale, self).__init__()
        self.add_module('conv2_', Conv2d(input_features, output_features * 4, kernel_size=3))  ####模型创新点
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))
        self.add_module('pixelshuffler', _PixelShuffler())  ####模型创新点


class Flatten(nn.Module):
    def forward(self, input):
        output = input.view(input.size(0), -1)  # 二次卷积的输出拉伸为一行
        return output


class Reshape(nn.Module):

    def forward(self, input):
        output = input.view(-1, 1024, 4, 4)  # channel * 4 * 4
        return output


class _PixelShuffler(nn.Module):  # 其功能是将filter的大小变为原来的1/4，让后让高h、宽w各变为原来的两倍
    def forward(self, input):
        batch_size, c, h, w = input.size()  # （-1,2048,4,4）
        rh, rw = (2, 2)
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        out = input.view(batch_size, rh, rw, oc, h, w)  # （-1，2，2，512，4,4）
        out = out.permute(0, 3, 4, 1, 5,
                          2).contiguous()  # permute()等改变形状    #contiguous()这个函数，把tensor变成在内存中连续分布的形式。 （-1，512，4，2，4,2）
        out = out.view(batch_size, oc, oh, ow)  # channel first     （-1，512，8，8）
        return out

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
            out = self.encoder(x)  # 编码      torch.Size([32, 512, 8, 8])
            out = self.decoder_A(out)  # 解码         类似反卷积还原成原来的图像  但是不是反卷积算法
        else:
            out = self.encoder(x)  # 编码      torch.Size([32, 512, 8, 8])
            out = self.decoder_B(out)  # 解码         类似反卷积还原成原来的图像  但是不是反卷积算法
        return out
