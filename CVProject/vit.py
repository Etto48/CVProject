import torch
from torch import nn
import torchvision
from tqdm.auto import tqdm

from CVProject.dataset import TextImageDataset
from CVProject.lambda_layer import LambdaLayer
from CVProject.pos_enc import LearnablePositionalEncoding, LearnablePositionalEncoding2d, PositionalEncoding, PositionalEncoding2d


class ViT(nn.Module):
    def __init__(self, embedding_dim: int, patch_size: int, dropout: float, transformer_depth: int, transformer_width: int, transformer_heads: int):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.augmentation = nn.Sequential(
            LambdaLayer(lambda x: (x + 1) / 2),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(
                10, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                fill=0.5),
            torchvision.transforms.RandomResizedCrop(
                128, scale=(0.8, 1.0), ratio=(0.75, 1.3)),
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            LambdaLayer(lambda x: x * 2 - 1)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(3, embedding_dim, kernel_size=patch_size, stride=patch_size // 2),
            nn.GELU(),
            LearnablePositionalEncoding2d(embedding_dim),
            nn.Flatten(start_dim=2),
            LambdaLayer(lambda x: x.permute(0, 2, 1))
        )
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
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
        if self.training:
            x = self.augmentation(x)
        x = self.conv(x)
        x = torch.cat([self.class_embedding.repeat(x.shape[0], 1, 1), x], dim=1)
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
