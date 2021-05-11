import torch.nn as nn
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch


# 定义3x3卷积核
def conv3x3(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


# 定义残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.convx = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.bnx = nn.BatchNorm2d(out_channels)

    def forward(self, x, stride):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(residual, stride)
        out += residual
        out = self.relu(out)
        return out

    def downsample(self, x, stride):
        if stride == 1:
            out = x
        else:
            out = self.convx(x)
            out = self.bnx(out)
        return out


# 定义残差神经网络
class ResNet(nn.Module):
    def __init__(self, output_length):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)  # (c,112,112)
        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (c,56,56)

        self.block1 = ResBlock(self.in_channels, self.in_channels, stride=1)
        self.block2 = ResBlock(self.in_channels, self.in_channels, stride=1)
        self.block3 = ResBlock(self.in_channels, self.in_channels * 2, stride=2)  # (2c,28,28)
        self.block4 = ResBlock(self.in_channels * 2, self.in_channels * 2, stride=1)
        self.block5 = ResBlock(self.in_channels * 2, self.in_channels * 4, stride=2)  # (4c,14,14)
        self.block6 = ResBlock(self.in_channels * 4, self.in_channels * 4, stride=1)
        self.block7 = ResBlock(self.in_channels * 4, self.in_channels * 8, stride=2)  # (8c,7,7)
        self.block8 = ResBlock(self.in_channels * 8, self.in_channels * 8, stride=1)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(self.in_channels * 8 * 1 * 1, output_length)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x, 1)
        x = self.block2(x, 1)
        x = self.block3(x, 2)
        x = self.block4(x, 1)
        x = self.block5(x, 2)
        x = self.block6(x, 1)
        x = self.block7(x, 2)
        x = self.block8(x, 1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 测试
# transform = T.Compose([
#          T.RandomResizedCrop(224),
#          T.ToTensor(),
# ])
#
# dataset = ImageFolder('./test/', transform=transform)
# loader = DataLoader(dataset=dataset, batch_size=10, shuffle=False)
# device = torch.device("cuda")
# model = ResNet(2).to(device)
#
# for i, (data, label) in enumerate(loader):
#     data = data.to(device)
#     print("data:", data.size())
#     outputs = model(data.float())
