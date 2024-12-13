import torch
from torch import nn
import torchvision
from tqdm.auto import tqdm

from CVProject.dataset import TextImageDataset
from CVProject.image_embedding import ImageEmbedding
from CVProject.lambda_layer import LambdaLayer
from CVProject.pos_enc import LearnablePositionalEncoding, LearnablePositionalEncoding2d, PositionalEncoding, PositionalEncoding2d


class ViT(nn.Module):
    def __init__(self, embedding_dim: int, conv_depth: int, conv_width: int, dropout: float, transformer_depth: int, transformer_width: int, transformer_heads: int):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.conv = ImageEmbedding(embedding_dim, conv_depth, conv_width, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_width,
                dropout=dropout,
                batch_first=True,
                activation="gelu"),
            num_layers=transformer_depth,
            enable_nested_tensor=True,
            norm=nn.LayerNorm(embedding_dim)
        )

        self.to(self.device)

    def forward(self, x):
        x = self.encode(x)
        return x

    def encode(self, x):
        x = self.conv(x)
        x = self.transformer(x)
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    valid = TextImageDataset.load_valid()
    vit = ViT(embedding_dim=256, patch_size=16, dropout=0.2,
              transformer_depth=1, transformer_width=256, transformer_heads=8)

    imgs = []
    l = 4
    for i, data in enumerate(valid):
        images, _, _ = data
        imgs.append(images)
        if i == l:
            break
    imgs = torch.stack(imgs)
    augmented_imgs = vit.augmentation(imgs)
    imgs = imgs.permute(0, 2, 3, 1)
    augmented_imgs = augmented_imgs.permute(0, 2, 3, 1)
    fig, ax = plt.subplots(2, l)
    for i in range(l):
        ax[0, i].imshow((imgs[i] + 1) / 2)
        ax[1, i].imshow((augmented_imgs[i] + 1) / 2)
    ax[0, -1].axis("off")
    ax[1, -1].axis("off")
    plt.show()
