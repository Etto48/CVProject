import torch
from torch import nn
import torchvision
from tqdm.auto import tqdm

from CVProject.dataset import TextImageDataset
from CVProject.lambda_layer import LambdaLayer
from CVProject.pos_enc import LearnablePositionalEncoding, LearnablePositionalEncoding2d, PositionalEncoding, PositionalEncoding2d


class ImageEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, conv_depth: int, conv_width: int, dropout: float):
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
        self.conv = nn.Sequential()
        self.conv_depth = conv_depth
        for i in range(self.conv_depth):
            in_c = 3 if i == 0 else conv_width
            out_c = conv_width if i < self.conv_depth - 1 else embedding_dim
            self.conv.append(nn.Conv2d(
                in_c,
                out_c,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                device=self.device))
            self.conv.append(nn.GELU())
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(nn.Dropout2d(dropout))
        self.conv.append(LearnablePositionalEncoding2d(embedding_dim))
        self.conv.append(nn.Flatten(start_dim=2))
        self.conv.append(LambdaLayer(lambda x: x.permute(0, 2, 1)))

    def forward(self, x):
        if self.training:
            x = self.augmentation(x)
        x = self.conv(x)
        return x
