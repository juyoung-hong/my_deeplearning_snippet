import torch
from torch import nn


class BasicConv2dBlock(nn.Module):
    """A Basic Convolution Block for building Convolutional Neural Networks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: nn.Module = None,
        act_layer: nn.Module = None,
        dropout: float = 0,
        **kwargs
    ) -> None:
        super().__init__()
        """Initialize a `BasicConv2dBlock` module.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param norm_layer: The normalization layer.
        :param act_layer: The non-linear activation function layer.
        :param dropout: The probability of random elemination of drop out.
        """
        conv_block = []
        conv_block.append(nn.Conv2d(in_channels, out_channels, **kwargs))
        if norm_layer is not None:
            conv_block.append(norm_layer)
        if act_layer is not None:
            conv_block.append(act_layer)
        if dropout != 0:
            conv_block.append(nn.Dropout(dropout))
        self.conv_block = nn.ModuleList(conv_block)

    def forward(self, x):
        """Perform a single forward pass through the Basic Conv Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        h = x
        for layer in self.conv_block:
            h = layer(h)
        return h
