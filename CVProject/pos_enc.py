import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, device=None):
        super().__init__()
        self.device = device
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)
        self.set_pe(max_len, d_model)

    def set_pe(self, t, d):
        position = torch.arange(t, device=self.device).view(1, -1, 1)
        div_term = torch.exp(torch.arange(0, d, 2, device=self.device) * -np.log(10000.0) / d).view(1, 1, -1)
        self.pe.data = torch.zeros(1, t, d, device=self.device)
        self.pe.data[:, :, 0::2] = torch.sin(position * div_term)
        self.pe.data[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, d = x.size()
        if t > self.pe.size(1):
            self.set_pe(t, d)

        x = x + self.pe[:, :t, :]
        return x
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pe = PositionalEncoding(512)
    x = torch.zeros(1, 100, 512)
    y = pe(x)
    plt.imshow(y[0].detach().numpy())
    plt.show()
        

