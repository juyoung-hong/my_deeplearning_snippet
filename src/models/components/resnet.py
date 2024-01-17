import torch
from torch import nn


class ResNet2DBlock(nn.Module):
    """A ResNet2DBlock for building ResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        shortcut: bool = False,
        norm_layer: nn.Module = None,
        act_layer: nn.Module = None,
        dropout: float = 0,
    ):
        """Initialize a `ResNet2DBlock` module.

        :param in_channels: The number of input channels.
        :param stride: The number of convolution stride.
        :param expansion: in_channels * expansion = out_channels.
        :param norm_layer: The normalization layer.
        :param act_layer: The non-linear activation function layer.
        :param dropout: The probability of random elemination of drop out.
        """
        super(ResNet2DBlock, self).__init__()
        self.act_layer = act_layer
        self.dropout = nn.Dropout(dropout) if dropout != 0 else None
        self.main_path = nn.Sequential(
            # downsampling
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=stride, padding=1
            ),
            norm_layer(in_channels),
            act_layer,
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels),
        )

        self.shortcut = None
        if stride != 1 or shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm_layer(out_channels),
            )

    def forward(self, x):
        """Perform a single forward pass through the ResNet2D Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        h = self.main_path(x)
        if self.shortcut is not None:
            h += self.shortcut(x)
        out = self.act_layer(h)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class BottleNeckBlock(nn.Module):
    """A BottleneckBlock for building ResNet."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        shortcut: bool = False,
        norm_layer: nn.Module = None,
        act_layer: nn.Module = None,
        dropout: float = 0,
    ):
        """Initialize a `BottleneckBlock` module.

        :param in_channels: The number of input channels.
        :param stride: The number of convolution stride.
        :param expansion: in_channels * expansion = out_channels.
        :param norm_layer: The normalization layer.
        :param act_layer: The non-linear activation function layer.
        :param dropout: The probability of random elemination of drop out.
        """
        super(BottleNeckBlock, self).__init__()
        self.act_layer = act_layer
        self.dropout = nn.Dropout(dropout) if dropout != 0 else None
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            norm_layer(mid_channels),
            act_layer,
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1
            ),
            norm_layer(mid_channels),
            act_layer,
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            norm_layer(out_channels),
        )

        self.shortcut = None
        if stride != 1 or shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm_layer(out_channels),
            )

    def forward(self, x):
        """Perform a single forward pass through the BottleNeck Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        h = self.main_path(x)
        if self.shortcut is not None:
            h += self.shortcut(x)
        out = self.act_layer(h)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ResNet50(nn.Module):
    """A ResNet-50 Network for computing predictions."""

    def __init__(
        self,
        num_classes: int = 200,
    ) -> None:
        """Initialize a `ResNet-50` module.

        :param num_classes: The number of output features of the final linear layer.
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            # Layer2
            BottleNeckBlock(
                64,
                64,
                256,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                256,
                64,
                256,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                256, 64, 256, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True)
            ),
            # Layer3
            BottleNeckBlock(
                256,
                128,
                512,
                2,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                512,
                128,
                512,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                512,
                128,
                512,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                512,
                128,
                512,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            # Layer4
            BottleNeckBlock(
                512,
                256,
                1024,
                2,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                1024,
                256,
                1024,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                1024,
                256,
                1024,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                1024,
                256,
                1024,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                1024,
                256,
                1024,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                1024,
                256,
                1024,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            # Layer5
            BottleNeckBlock(
                1024,
                512,
                2048,
                2,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                2048,
                512,
                2048,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
            BottleNeckBlock(
                2048,
                512,
                2048,
                1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU(inplace=True),
            ),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)
        self.init_param()  # initialize parameters

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # init conv
                nn.init.kaiming_normal_(m.weight)  # He init (for ReLU)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):  # init BN
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # lnit linear
                nn.init.kaiming_normal_(m.weight)  # He init (for ReLU)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        h = self.feature_extractor(x)
        h = self.avgpool(h)
        # (batch, 1, width, height) -> (batch, 1*width*height)
        h = h.view(batch_size, -1)
        out = self.classifier(h)

        return out


if __name__ == "__main__":
    _ = ResNet50()
