from torch.nn import (Linear, Conv2d, BatchNorm2d, BatchNorm1d, ReLU,
                      Dropout, Module, MaxPool2d, Dropout, Sequential)
import torch.nn as nn
import torch


# from torchkeras import summary


def conv3x3(in_channel, out_channel, stride=1):
    '''3x3 convolution with padding'''
    return Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channel, out_channel, stride=1):
    '''1x1 convolution with padding'''
    return Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_channel, channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_channel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(channel)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_channel, channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm2d(channel)
        self.conv2 = Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(channel)
        self.conv3 = Conv2d(channel, channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm2d(channel * self.expansion)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResNet(Module):

    def __init__(self, block, layers, feature_dim=512, drop_ratio=0.4, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.output_layer = nn.Sequential(BatchNorm2d(512 * block.expansion),
                                          Dropout(drop_ratio),
                                          Flatten(),
                                          Linear(512 * block.expansion * 7 * 7, feature_dim),
                                          BatchNorm1d(feature_dim))

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = Sequential(
                conv1x1(self.in_channel, channel * block.expansion, stride),
                BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample))
        self.in_channel = channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.output_layer(x)

        return x


def ResNet18(feature_dim):
    model = ResNet(BasicBlock, [2, 2, 2, 2], feature_dim)
    return model


def ResNet34(feature_dim):
    model = ResNet(BasicBlock, [3, 4, 6, 3], feature_dim)
    return model


def ResNet50(feature_dim):
    model = ResNet(Bottleneck, [3, 4, 14, 3], feature_dim)
    return model


def ResNet101(feature_dim):
    model = ResNet(Bottleneck, [3, 4, 23, 3], feature_dim)
    return model


def ResNet152(feature_dim):
    model = ResNet(Bottleneck, [3, 8, 36, 3], feature_dim)
    return model


if __name__ == '__main__':
    input = torch.Tensor(2, 3, 112, 112)
    net = ResNet50()
    # print(net)
    # summary(net, input_shape=(3, 112, 112))
    x = net(input)
    print(x.shape)
