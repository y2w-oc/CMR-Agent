import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, inchannel=3, outchannel=3, stride=1):
        super(ResidualBlock, self).__init__()
        assert (stride == 1 or stride == 2)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 3, stride, 1),
            nn.BatchNorm2d(inchannel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(inchannel, outchannel, 3, 1, 1),
            nn.BatchNorm2d(outchannel)
        )
        self.final_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if inchannel != outchannel:
            if stride == 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, 1, 1, 0),
                    nn.BatchNorm2d(outchannel),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, 3, 2, 1),
                    nn.BatchNorm2d(outchannel),
                )
        else:
            if stride == 1:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, 3, 2, 1),
                    nn.BatchNorm2d(outchannel),
                )

    def forward(self, x):
        out = self.conv_layers(x) + self.shortcut(x)
        return self.final_relu(out)


class MiniResNet(nn.Module):
    """
    Implementation of a simple ResNet. The output scale will be downsampled to 1/4
    """
    def __init__(self, inchannel=3, outchannel=3):
        super(MiniResNet, self).__init__()

        self.residual_learning = nn.ModuleList()
        self.residual_learning.append(ResidualBlock(inchannel, outchannel, 1))
        self.residual_learning.append(ResidualBlock(outchannel, outchannel, 1))
        self.residual_learning.append(ResidualBlock(outchannel, outchannel, 2))
        self.residual_learning.append(ResidualBlock(outchannel, outchannel, 1))
        self.residual_learning.append(ResidualBlock(outchannel, outchannel, 2))
        self.residual_learning.append(ResidualBlock(outchannel, outchannel, 1))

    def forward(self, x):
        x = self.residual_learning[0](x)
        img_feat_0 = self.residual_learning[1](x)
        x = self.residual_learning[2](img_feat_0)
        img_feat_1 = self.residual_learning[3](x)
        x = self.residual_learning[4](img_feat_1)
        img_feat_2 = self.residual_learning[5](x)
        return img_feat_2, img_feat_1, img_feat_0