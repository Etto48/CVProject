import torch
from torch import nn

from CVProject.pos_enc import PositionalEncoding

class ViT(nn.Module):
    def __init__(self, conv_depth: int, embedding_dim: int, filter_size: int, dropout: float, transformer_depth: int, transformer_width: int, transformer_heads: int):
        super().__init__()
        self.conv = nn.Sequential()

        for i in range(conv_depth):
            in_channels = 3 if i == 0 else embedding_dim
            self.conv.append(
                nn.Conv2d(
                    in_channels, 
                    embedding_dim, 
                    filter_size, 
                    padding=filter_size // 2, 
                    padding_mode="reflect"))
            self.conv.append(nn.ReLU())
            self.conv.append(nn.Dropout2d(dropout))
            if i < conv_depth - 1:
                self.conv.append(nn.MaxPool2d(4))
        self.conv.append(nn.Flatten(start_dim=2))
        self.pos_enc = PositionalEncoding(embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_width,
                dropout=dropout,
                batch_first=True),
                num_layers=transformer_depth,
                enable_nested_tensor=True,
                norm=nn.LayerNorm(embedding_dim)
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return x
