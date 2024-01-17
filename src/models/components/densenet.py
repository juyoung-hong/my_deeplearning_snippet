import torch
from torch import nn


class BottleNeckBlock(nn.Module):
    """A BottleneckBlock for building DenseNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # growth rate
        stride: int = 1,
        norm_layer: nn.Module = None,
        act_layer: nn.Module = None,
        dropout: float = 0,
    ):
        """Initialize a `BottleneckBlock` module.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param stride: The number of convolution stride.
        :param norm_layer: The normalization layer.
        :param act_layer: The non-linear activation function layer.
        :param dropout: The probability of random elemination of drop out.
        """
        super(BottleNeckBlock, self).__init__()
        mid_channels = out_channels * 4

        self.act_layer = act_layer
        self.dropout = nn.Dropout(dropout) if dropout != 0 else None
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            act_layer,
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            norm_layer(mid_channels),
            act_layer,
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
        )

    def forward(self, x):
        """Perform a single forward pass through the BottleNeck Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        h = self.main_path(x)
        out = torch.cat([x, h], 1)
        return out


class DenseBlock(nn.Module):
    """A DenseBlock for building DenseNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # growth rate
        n_layers,
        norm_layer: nn.Module = None,
        act_layer: nn.Module = None,
        dropout: float = 0,
    ):
        """Initialize a `DenseBlock` module.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param n_layers: The number of repeated bottleneck layers.
        :param norm_layer: The normalization layer.
        :param act_layer: The non-linear activation function layer.
        :param dropout: The probability of random elemination of drop out.
        """
        super(DenseBlock, self).__init__()
        dense_block = []
        for i in range(n_layers):
            dense_block.append(
                BottleNeckBlock(
                    in_channels + i * out_channels,
                    out_channels,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    dropout=dropout,
                )
            )
        self.dense_block = nn.ModuleList(dense_block)

    def forward(self, x):
        """Perform a single forward pass through the BottleNeck Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        h = x
        for layer in self.dense_block:
            h = layer(h)
        return h


class TransitionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # growth rate
        norm_layer: nn.Module = None,
        act_layer: nn.Module = None,
        dropout: float = 0,
    ):
        """Initialize a `TransitionBlock` module.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param n_layers: The number of repeated bottleneck layers.
        :param norm_layer: The normalization layer.
        :param act_layer: The non-linear activation function layer.
        :param dropout: The probability of random elemination of drop out.
        """
        super(TransitionBlock, self).__init__()
        transition_block = [
            norm_layer(in_channels),
            act_layer,
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.AvgPool2d(2, 2, ceil_mode=True),
        ]

        self.transition_block = nn.ModuleList(transition_block)

    def forward(self, x):
        """Perform a single forward pass through the BottleNeck Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        h = x
        for layer in self.transition_block:
            h = layer(h)
        return h


class DenseNet121(nn.Module):
    """A DenseNet-121 Network for computing predictions."""

    def __init__(
        self,
        growth_rate: int = 32,
        compression: float = 0.5,
        num_classes: int = 200,
    ) -> None:
        """Initialize a `DenseNet-121` module.

        :param growth_rate: The number of output feature channels of the one layer.
        :param compression: compression factor of transition layers.
        :param num_classes: The number of output features of the final linear layer.
        """
        super().__init__()
        self.n_layers = [6, 12, 24, 16]
        self.feature_extractor = [
            nn.Conv2d(3, 2 * growth_rate, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True),
        ]
        n_channels = int(2 * growth_rate)
        for n in self.n_layers:
            self.feature_extractor.append(
                DenseBlock(
                    n_channels,
                    growth_rate,
                    n,
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.ReLU(inplace=True),
                )
            )
            self.feature_extractor.append(
                TransitionBlock(
                    in_channels=n_channels + n * growth_rate,
                    out_channels=int(n_channels + n * growth_rate * compression),
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.ReLU(inplace=True),
                )
            )
            n_channels = int(n_channels + n * growth_rate * compression)
        self.feature_extractor = nn.ModuleList(self.feature_extractor)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_channels, num_classes)
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

        h = x
        for layer in self.feature_extractor:
            h = layer(h)
        h = self.avgpool(h)
        # (batch, 1, width, height) -> (batch, 1*width*height)
        h = h.view(batch_size, -1)
        out = self.classifier(h)

        return out


if __name__ == "__main__":
    _ = DenseNet121()
