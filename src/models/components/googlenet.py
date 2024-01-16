import torch
from torch import nn

from .basic import BasicConv2dBlock


class CBRBlock(nn.Module):
    """A Convolution BatchNorm ReLU Block"""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        """Initialize a `CBR` module.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        """
        super().__init__()
        self.block = BasicConv2dBlock(
            in_channels,
            out_channels,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=nn.ReLU(inplace=True),
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the CBR Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        return self.block(x)


class InceptionBlock(nn.Module):
    """A Inception Block for building GoogLeNet."""

    def __init__(
        self,
        in_channels: int,
        n1: int,
        n3_reduce: int,
        n3: int,
        n5_reduce: int,
        n5: int,
        pool_proj: int,
    ) -> None:
        """Initialize a `Inception Block` module.

        :param in_channels: The number of input channels.
        :param n1: The number of 1x1 filters.
        :param n3_reduce: The number of 1x1 filters in the reduction layer used before the 3x3 convolutions.
        :param n3: The number of 3x3 filters.
        :param n5_reduce: The number of 1x1 filters in the reduction layer used before the 5x5 convolutions.
        :param n5: The number of 5x5 filters.
        :param pool_proj: The number of channels for MaxPooling output.

        """
        super().__init__()
        self.branch1 = CBRBlock(
            in_channels,
            n1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.branch2 = nn.Sequential(
            CBRBlock(
                in_channels,
                n3_reduce,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            CBRBlock(
                n3_reduce,
                n3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.branch3 = nn.Sequential(
            CBRBlock(
                in_channels,
                n5_reduce,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            CBRBlock(
                n5_reduce,
                n5,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            CBRBlock(
                in_channels,
                pool_proj,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the Inception Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class AuxiliaryClassifier(nn.Module):
    """An auxiliary classifier Block for building GoogLeNet."""

    def __init__(self, in_channels, num_classes) -> None:
        """Initialize a `auxiliary classifier Block` module.

        :param in_channels: The number of input channels.
        :param num_classes: The number of output features of the final linear layer.

        """
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0),
        )
        self.net = nn.Sequential(
            nn.Linear(128, num_classes),  # (2048, 1024)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(num_classes, num_classes),  # (1024, n_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the Auxiliary Classifier Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        h = self.conv_net(x)
        # (batch, 1, width, height) -> (batch, 1*width*height)
        h = h.view(batch_size, -1)
        out = self.net(h)
        return out


class GoogLeNet(nn.Module):
    """A GoogLeNet for computing predictions."""

    def __init__(
        self,
        num_classes: int = 200,
    ) -> None:
        """Initialize a `GoogLeNet` module.

        :param num_classes: The number of output features of the final linear layer.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            CBRBlock(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            CBRBlock(
                in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0
            ),
            CBRBlock(
                in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.a3 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.a4 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a5 = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, num_classes)

        self.aux1 = AuxiliaryClassifier(512, num_classes)
        self.aux2 = AuxiliaryClassifier(528, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        h = self.conv1(x)
        h = self.a3(h)
        h = self.b3(h)
        h = self.maxpool3(h)
        h = self.a4(h)
        aux1 = self.aux1(h)

        h = self.b4(h)
        h = self.c4(h)
        h = self.d4(h)
        aux2 = self.aux2(h)

        h = self.e4(h)
        h = self.maxpool4(h)
        h = self.a5(h)
        h = self.b5(h)
        h = self.avgpool(h)

        batch_size, channels, width, height = h.size()
        # (batch, 1, width, height) -> (batch, 1*width*height)
        h = h.view(batch_size, -1)
        h = self.linear(h)
        out = self.dropout(h)

        return (out, aux1, aux2)


if __name__ == "__main__":
    _ = GoogLeNet()
