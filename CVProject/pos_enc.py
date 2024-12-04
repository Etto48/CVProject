import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = None

    def set_pe(self, t, d, device):
        position = torch.arange(t, device=device).view(1, -1, 1)
        div_term = torch.exp(torch.arange(0, d, 2, device=device) * -np.log(10000.0) / d).view(1, 1, -1)
        self.pe = torch.zeros(1, t, d, device=device)
        self.pe[:, :, 0::2] = torch.sin(position * div_term)
        self.pe[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, d = x.size()
        if self.pe is None or t > self.pe.size(1) or d != self.pe.size(2) or self.pe.device != x.device:
            self.set_pe(t, d, x.device)

        x = x + self.pe[:, :t, :]
        return x

class PositionalEncoding2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = None

    def set_pe(self, h, w, device=None):
        self.pe = torch.zeros(1, 3, h, w, device=device, requires_grad=False)
        h_position = torch.arange(h, device=device).view(1, 1, h, 1) / (h - 1) * 2 - 1
        w_position = torch.arange(w, device=device).view(1, 1, 1, w) / (w - 1) * 2 - 1
        central_heatmap = torch.zeros(1, 1, h, w, device=device)
        for i in range(0, h):
            for j in range(0, w):
                central_heatmap[:, :, i, j] =  1-((i/(h-1)-0.5)**2 + (j/(w-1)-0.5)**2)*4
        self.pe[:, 0, :, :] = central_heatmap
        self.pe[:, 1, :, :] = h_position
        self.pe[:, 2, :, :] = w_position
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        if self.pe is None or self.pe.size(2) != h or self.pe.size(3) != w or self.pe.device != x.device:
            self.set_pe(h, w, x.device)
        
        x[:, :3] = x[:, :3] + self.pe
        return x
        
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, t, d):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, t, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, d = x.size()
        x = x + self.pe[:, :t, :]
        return x
    
class LearnablePositionalEncoding2d(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Conv2d(3, d, kernel_size=1)
        self.pe = PositionalEncoding2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3], device=x.device)
        z = self.pe(z)
        z = self.linear(z)
        return x + z

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pe = PositionalEncoding()
    x = torch.zeros(1, 512, 128)
    y = pe(x)
    plt.subplot(2, 1, 1)
    plt.title("Positional Encoding")
    plt.imshow(y[0].T.detach().numpy())
    plt.colorbar()
    plt.xlabel("Time-step")
    plt.ylabel("Feature")

    pe = LearnablePositionalEncoding2d(128)
    x = torch.zeros(1, 128, 16, 16)
    y = pe(x)
    plt.subplot(2, 4, 5)
    plt.title("Positional Encoding 2D")
    plt.imshow(y.view(128, -1).detach().numpy())
    plt.colorbar()
    plt.subplot(2, 4, 6)
    plt.imshow(y[0, 0, :, :].detach().numpy())
    plt.colorbar()
    plt.subplot(2, 4, 7)
    plt.imshow(y[0, 1, :, :].detach().numpy())
    plt.colorbar()
    plt.subplot(2, 4, 8)
    plt.imshow(y[0, 2, :, :].detach().numpy())
    plt.colorbar()
    plt.show()

        

