import torch
from einops import repeat
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from .attention import TransformerEncoderBlock


class ImageEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 64,
        patch_size: int = 16,
        emb_dim=16 * 16 * 3,
    ) -> None:
        """Initialize a `Image Embedding Block` module.

        :param in_channels: The number of channels in the hidden sates.
        :param img_size: The size of Image (width or height).
        :param patch_size: The size of patch (width or height).
        :param emb_dim: The size of latent vector.
        """
        super().__init__()

        self.rearrange = Rearrange(
            "b c (num_w p1) (num_h p2) -> b (num_w num_h) (p1 p2 c) ",
            p1=patch_size,
            p2=patch_size,
        )
        self.linear = nn.Linear(in_channels * patch_size * patch_size, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        n_patches = img_size * img_size // patch_size**2
        self.pos_emb = nn.Parameter(torch.randn(n_patches + 1, emb_dim))

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        """Perform a single forward pass through the Image Embedding.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channel, width, height = x.size()

        x = self.rearrange(x)  # flatten patches
        x = self.linear(x)  # embedded patches

        c = repeat(self.cls_token, "() n d -> b n d", b=batch_size)
        x = torch.cat((c, x), dim=1)  # add learnable class token
        x = x + self.pos_emb  # learnable positional embedding
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 64,
        patch_size: int = 16,
        emb_dim=768,
        n_encoder: int = 12,
        n_head: int = 12,
        expansion: int = 4,
        dropout: float = 0.2,
        num_classes: int = 200,
    ) -> None:
        """Initialize a `Vision Transformer` module.

        :param in_channels: The number of channels in the hidden sates.
        :param img_size: The size of Image (width or height).
        :param patch_size: The size of patch (width or height).
        :param emb_dim: The size of latent vector.
        :param n_encoder: The number of Transformer Encoder Blocks.
        :param n_head: The number of Multi-Head Attention Heads.
        :param expansion: The number of channels that will be multiplied with in_channels.
        :param dropout: The probability of random elemination of drop out.
        :param num_classes: The number of output features of the final linear layer.
        """
        super().__init__()

        self.image_emb = ImageEmbedding(in_channels, img_size, patch_size, emb_dim)
        self.transformer_encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(emb_dim, n_head, expansion, nn.GELU(), dropout)
                for _ in range(n_encoder)
            ]
        )

        self.reduce_layer = Reduce("b n e -> b e", reduction="mean")
        self.normalization = nn.LayerNorm(emb_dim)
        self.classification_head = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        """Perform a single forward pass through the Image Embedding.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # Image Embedding (learnable class token, positional embedding)
        x = self.image_emb(x)

        # Transformer Encoder
        attentions = []
        for encoder in self.transformer_encoders:
            x, att = encoder(x)
            attentions.append(att)

        # Classification Head
        x = self.reduce_layer(x)
        x = self.normalization(x)
        x = self.classification_head(x)

        return x, attentions
