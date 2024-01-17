import torch
from torch import nn

from .basic import BasicConv2dBlock


class VGGBlock(nn.Module):
    """A VGG Block for building VGG Networks."""

    def __init__(
        self, in_channels: int, out_channels: int, repeat: int, pool: bool = False
    ) -> None:
        """Initialize a `VGGBlock` module.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param repeat: The number of repeat of basic convolution blocks.
        :param pool: The flag of Maxpooling.
        """
        super().__init__()
        vgg_block = []
        for i in range(repeat):
            vgg_block.append(
                BasicConv2dBlock(
                    in_channels,
                    out_channels,
                    norm_layer=nn.BatchNorm2d(out_channels),
                    act_layer=nn.ReLU(inplace=True),
                    kernel_size=3,
                    padding=1,
                )
            )
            in_channels = out_channels
        if pool:
            vgg_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vgg_block = nn.ModuleList(vgg_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the VGG Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        h = x
        for layer in self.vgg_block:
            h = layer(h)
        return h


class VGG19(nn.Module):
    """A VGG-19 Network for computing predictions."""

    def __init__(
        self,
        num_classes: int = 200,
    ) -> None:
        """Initialize a `VGG-19` module.

        :param num_classes: The number of output features of the final linear layer.
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(
            VGGBlock(3, 64, 2, True),
            VGGBlock(64, 128, 2, True),
            VGGBlock(128, 256, 4, True),
            VGGBlock(256, 512, 4, True),
            VGGBlock(512, 512, 4, True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
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
    _ = VGG19()
