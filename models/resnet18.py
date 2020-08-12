import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_channels, out_channels, num_block, stride=1):
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for i in range(num_block - 1):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, num_block=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, num_block=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, num_block=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, num_block=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        x = self.conv1(x)  # 1/2
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)  # 1/4

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8, chs=128
        feat16 = self.layer3(feat8)  # 1/16, chs=256
        feat32 = self.layer4(feat16)  # 1/32, chs=512

        x_avg = self.avgpool(feat32)
        x_avg = torch.flatten(x_avg, 1)
        oup = self.fc(x_avg)
        return oup


if __name__ == '__main__':
    inp = torch.randn(4, 3, 360, 640)
    net = Resnet18()
    oup = net(inp)
    print(oup.shape)


