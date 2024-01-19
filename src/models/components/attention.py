# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_K = K.size()[-1]  # key dimension
    scores = Q.matmul(K.transpose(-2, -1)) * (d_K**-0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = F.softmax(scores, dim=-1)
    out = attention.matmul(V)
    return out, attention


class AttentionBlock(nn.Module):
    """A Basic Attention Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_channels: Optional[int] = None,
        attention_strategy: Callable = scaled_dot_product_attention,
    ) -> None:
        super().__init__()
        """Initialize a `AttentionBlock` module.

        :param in_channels: The number of channels in the hidden sates.
        :param out_channels: The number of channels in the query, key, value.
        :param encoder_channels: The number of channels in the encoder_hidden_states. If not given, defaults to `embedding_dim`.
        :param attention_strategy: The operations for attention. defaults 'scaled dot product attention'.
        """
        self.attention_strategy = attention_strategy
        encoder_channels = (
            encoder_channels if encoder_channels is not None else in_channels
        )
        self.to_q = nn.Linear(in_channels, out_channels)
        self.to_k = nn.Linear(encoder_channels, out_channels)
        self.to_v = nn.Linear(encoder_channels, out_channels)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Perform a single forward pass through the Attention Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        Q = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        K = self.to_k(encoder_hidden_states)
        V = self.to_v(encoder_hidden_states)

        hidden_states, attention = self.attention_strategy(Q, K, V, attention_mask)

        return hidden_states, attention


class MultiHeadAttentionBlock(nn.Module):
    """A Multi-Head Attention Block"""

    def __init__(
        self,
        in_channels: int,
        encoder_channels: Optional[int] = None,
        n_head: int = 8,
        act_layer: nn.Module = nn.ReLU(inplace=True),
        attention_strategy: Callable = scaled_dot_product_attention,
    ) -> None:
        super().__init__()
        """Initialize a `Multi-Head Attention Block` module.

        :param in_channels: The number of channels in the hidden sates.
        :param encoder_channels: The number of channels in the encoder_hidden_states. If not given, defaults to `embedding_dim`.
        :param n_head: number of attention heads.
        :param act_layer: The non-linear activation function layer.
        :param attention_strategy: The operations for attention. defaults 'scaled dot product attention'.
        """
        if (in_channels % n_head) != 0:
            raise ValueError(
                f"in_channels({in_channels}) should be divisible by n_head({n_head})"
            )
        self.n_head = n_head
        self.head_dim = in_channels // n_head
        self.attention_strategy = attention_strategy
        self.out_channels = in_channels

        encoder_channels = (
            encoder_channels if encoder_channels is not None else in_channels
        )
        self.to_q = nn.Linear(in_channels, self.out_channels)
        self.to_k = nn.Linear(encoder_channels, self.out_channels)
        self.to_v = nn.Linear(encoder_channels, self.out_channels)
        self.to_out = nn.Linear(self.out_channels, self.out_channels)
        self.act_layer = act_layer

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Perform a single forward pass through the Attention Block.

        :param hidden_states: The input tensor.
        :param encoder_hidden_states: The input tensor from encoder block.
        :param attention_mask: The attention_mask.
        :return: A tensor of predictions.
        """
        batch_size = hidden_states.shape[0]

        Q = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        K = self.to_k(encoder_hidden_states)
        V = self.to_v(encoder_hidden_states)

        # Multi-head split of Q, K, V [n_batch, n_head, n_Q, d_head]
        Q_split = Q.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K_split = K.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V_split = V.view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # hidden_states: [n_batch, n_head, n_Q, d_head]
        # attention: [n_batch, n_head, n_Q, n_K]
        hidden_states, attention = self.attention_strategy(
            Q_split, K_split, V_split, attention_mask
        )

        # reshape to [n_batch, n_Q, n_head, d_head]
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        # reshape to [n_batch, n_Q, d_out]
        hidden_states = hidden_states.view(batch_size, -1, self.out_channels)

        # Feed Forward
        hidden_states = self.to_out(hidden_states)

        return hidden_states, attention


class FeedFowardBlock(nn.Module):
    """A FeedFoward Block"""

    def __init__(
        self,
        in_channels: int,
        expansion: int = 4,
        act_layer: nn.Module = nn.GELU(),
        dropout: Optional[float] = 0.2,
    ) -> None:
        super().__init__()
        """Initialize a `Feed Foward Block` module.

        :param in_channels: The number of channels in the hidden sates.
        :param expansion: The number of channels that will be multiplied with in_channels.
        :param act_layer: The non-linear activation function layer.
        :param dropout: The probability of random elemination of drop out.
        """
        net = []
        net.append(nn.Linear(in_channels, in_channels * expansion))
        net.append(act_layer)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
        net.append(nn.Linear(in_channels * expansion, in_channels))
        self.net = nn.ModuleList(net)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ) -> torch.Tensor:
        """Perform a single forward pass through the Feed Foward Block.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        for layer in self.net:
            hidden_states = layer(hidden_states)

        return hidden_states


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_head: int = 8,
        expansion: int = 4,
        act_layer: nn.Module = nn.GELU(),
        dropout: Optional[float] = 0.2,
    ) -> None:
        """Initialize a `Transformer Encoder Block` module.

        :param in_channels: The number of channels in the hidden sates.
        :param n_head: The number of Multi-Head Attention Heads.
        :param expansion: The number of channels that will be multiplied with in_channels.
        :param act_layer: The non-linear activation function layer.
        :param dropout: The probability of random elemination of drop out.
        """
        super().__init__()

        self.mha = MultiHeadAttentionBlock(in_channels, None, n_head)
        self.norm1 = nn.LayerNorm(in_channels)

        self.ff = FeedFowardBlock(in_channels, expansion, act_layer, dropout)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Perform a single forward pass through the Transformer Encoder Block.

        :param hidden_states: The input tensor.
        :param attention_mask: The attention_mask.
        :return: A tensor of predictions.
        """
        # Self-Attention
        attn_outs, attention = self.mha(hidden_states, None, attention_mask)
        hidden_states = hidden_states + attn_outs
        hidden_states = self.norm1(hidden_states)

        # FeedForward
        ff_outs = self.ff(hidden_states)
        hidden_states = hidden_states + ff_outs
        hidden_states = self.norm2(hidden_states)

        return hidden_states, attention


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoder_channels: int,
        n_head: int = 8,
        expansion: int = 4,
        act_layer: nn.Module = nn.GELU(),
        dropout: Optional[float] = 0.2,
    ) -> None:
        """Initialize a `Transformer Decoder Block` module.

        :param in_channels: The number of channels in the hidden sates.
        :param encoder_channels: The number of channels in the encoder hidden sates.
        :param n_head: The number of Multi-Head Attention Heads.
        :param expansion: The number of channels that will be multiplied with in_channels.
        :param act_layer: The non-linear activation function layer.
        :param dropout: The probability of random elemination of drop out.
        """
        super().__init__()

        self.mha = MultiHeadAttentionBlock(in_channels, None, n_head)
        self.norm1 = nn.LayerNorm(in_channels)

        self.mha = MultiHeadAttentionBlock(in_channels, encoder_channels, n_head)
        self.norm2 = nn.LayerNorm(in_channels)

        self.ff = FeedFowardBlock(in_channels, expansion, act_layer, dropout)
        self.norm3 = nn.LayerNorm(in_channels)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Perform a single forward pass through the Transformer Decoder Block.

        :param hidden_states: The input tensor.
        :param encoder_hidden_states: The input tensor from encoder block.
        :param attention_mask: The attention_mask.
        :param encoder attention_mask: The attention_mask from encoder block.
        :return: A tensor of predictions.
        """
        # Self-Attention
        attn_outs, attention = self.mha(hidden_states, None, attention_mask)
        hidden_states = hidden_states + attn_outs
        hidden_states = self.norm1(hidden_states)

        # Cross-Attention
        attn_outs, attention = self.mha(
            hidden_states, encoder_hidden_states, encoder_attention_mask
        )
        hidden_states = hidden_states + attn_outs
        hidden_states = self.norm2(hidden_states)

        # FeedForward
        ff_outs = self.ff(hidden_states)
        hidden_states = hidden_states + ff_outs
        hidden_states = self.norm3(hidden_states)

        return hidden_states, attention
