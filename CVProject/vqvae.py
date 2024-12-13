import numpy as np
import torch
from torch import nn
import torchvision
from tqdm import tqdm
import vector_quantize_pytorch as vqp

from CVProject.dataset import TextImageDataset
from CVProject.lambda_layer import LambdaLayer


class ResBlock(nn.Module):
    def __init__(self, inout_c: int, kernel_size: int, dropout: float, device: torch.device):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                inout_c, 
                inout_c, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                padding_mode="reflect", 
                device=device),
            nn.BatchNorm2d(inout_c, device=device) if dropout == 0 else nn.Identity(),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                inout_c, 
                inout_c, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                padding_mode="reflect", 
                device=device),
            nn.BatchNorm2d(inout_c, device=device) if dropout == 0 else nn.Identity(),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                inout_c, 
                inout_c, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                padding_mode="reflect", 
                device=device),
            nn.Dropout2d(dropout),
        )
        self.output = nn.Sequential(
            nn.InstanceNorm2d(inout_c, device=device) if dropout != 0 else nn.Identity(),
            nn.BatchNorm2d(inout_c, device=device) if dropout == 0 else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.output(self.block(x) + x)

class VQVAE(nn.Module):
    def __init__(self, conv_depth: int, conv_width: int, codebook_size: int, dropout: float):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.augmentation = nn.Sequential(
            LambdaLayer(lambda x: (x + 1) / 2),
            torchvision.transforms.RandomRotation(
                10, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                fill=0.5),
            torchvision.transforms.RandomResizedCrop(
                128, scale=(0.8, 1.0), ratio=(0.75, 1.3)),
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            LambdaLayer(lambda x: x * 2 - 1)
        )
        self.encoder = nn.Sequential()
        filter_size = 3
        #filters_size = [1 + 2*(i+1) for i in range(conv_depth)]
        #filters_size.reverse()
        self.encoder.append(nn.Conv2d(3, conv_width, filter_size, padding=filter_size//2, padding_mode="reflect", device=self.device))
        if dropout == 0:
            self.encoder.append(nn.BatchNorm2d(conv_width, device=self.device))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Dropout2d(dropout))
        for i in range(conv_depth):
            self.encoder.append(ResBlock(conv_width, filter_size, dropout=dropout, device=self.device))
            self.encoder.append(nn.MaxPool2d(2))

        self.vq = vqp.VectorQuantize(
            conv_width, 
            codebook_size, 
            decay=0.7, 
            commitment_weight=1,
            use_cosine_sim=True,
        ).to(self.device)

        self.decoder = nn.Sequential()
        #filters_size.reverse()
        for i in range(conv_depth):
            self.decoder.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            if dropout == 0:
                self.decoder.append(nn.BatchNorm2d(conv_width, device=self.device))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Dropout2d(dropout))
            self.decoder.append(ResBlock(conv_width, filter_size, dropout=dropout, device=self.device))
        self.decoder.append(nn.Conv2d(conv_width, 3, filter_size, padding=filter_size//2, padding_mode="reflect", device=self.device))
        self.decoder.append(nn.Tanh())

    def forward(self, x):
        if self.training:
            x = self.augmentation(x)
        x_latent, (h, w) = self.encode(x)
        q, c, vq_loss = self.quantize(x_latent)
        x = self.decode(q, (h, w))
        return x, q, c, vq_loss
    
    def encode(self, x):
        x: torch.Tensor = self.encoder(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).permute(0, 2, 1)
        return x, (h, w)
    
    def decode(self, q, latent_shape):
        h, w = latent_shape
        b, n, c = q.shape
        assert n == h * w, "Invalid input shape"
        q = q.permute(0, 2, 1).reshape(-1, c, h, w)
        x = self.decoder(q)
        return x

    def quantize(self, x):
        q, c, vq_loss = self.vq(x)
        return q, c, vq_loss

    def loss(self, output_image: torch.Tensor, expected_image: torch.Tensor):
        l = nn.functional.l1_loss(output_image, expected_image)
        return l

    def fit(self, train: TextImageDataset, valid: TextImageDataset, epochs: int, batch_size: int = 4, lr: float = 1e-4):
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, 
            sampler=torch.utils.data.RandomSampler(train, replacement=True, num_samples=batch_size * 100),
            collate_fn=TextImageDataset.collate_fn)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, 
            sampler=torch.utils.data.RandomSampler(valid, replacement=True, num_samples=batch_size * 50),
            collate_fn=TextImageDataset.collate_fn)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_loss = float("inf")
        best_model = None
        patience = 3
        threshold = 0.001
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.train()
            avg_reconstruction_loss = 0
            avg_commitment_loss = 0
            batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (train)")
            for i, data in enumerate(batches):
                images, _, _, _ = data
                images = images.to(self.device)
                optimizer.zero_grad()
                output_image, quantized_latent, _, vq_loss = self(images)
                mse = self.loss(output_image, images)
                loss = mse + vq_loss
                avg_reconstruction_loss += mse.item()
                avg_commitment_loss += vq_loss.item()
                batches.set_postfix(
                    r=avg_reconstruction_loss / (i + 1),
                    c=avg_commitment_loss / (i + 1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1)
                optimizer.step()
            self.eval()
            avg_reconstruction_loss = 0
            avg_commitment_loss = 0
            batches = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} (valid)")
            with torch.no_grad():
                for i, data in enumerate(batches):
                    images, _, _, _ = data
                    images = images.to(self.device)
                    output_image, quantized_latent, _, vq_loss = self(images)
                    mse = self.loss(output_image, images)
                    avg_reconstruction_loss += mse.item()
                    avg_commitment_loss += vq_loss.item()
                    batches.set_postfix(
                        r=avg_reconstruction_loss / (i + 1),
                        c=avg_commitment_loss / (i + 1))
            avg_reconstruction_loss /= (i + 1)
            if avg_reconstruction_loss < best_loss - threshold:
                best_loss = avg_reconstruction_loss
                best_model = self.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        self.load_state_dict(best_model)
                    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train = TextImageDataset.load_train()
    valid = TextImageDataset.load_valid()
    model = VQVAE(conv_depth=2, conv_width=256, codebook_size=512, dropout=0)
    try:
        model.fit(train, valid, epochs=100)
    except KeyboardInterrupt:
        pass

    try:
        l = 4
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=l, shuffle=True, collate_fn=TextImageDataset.collate_fn)
        samples, _, _, _ = next(iter(valid_loader))
        samples = samples.to(model.device)
        with torch.no_grad():
            output, q, c, _ = model(samples)
        samples = samples.cpu()
        output = output.cpu()
        for i in range(l):
            plt.subplot(2, l, i + 1)
            plt.imshow((samples[i].permute(1, 2, 0) + 1) / 2)
            plt.axis("off")
            plt.subplot(2, l, i + l + 1)
            plt.imshow((output[i].permute(1, 2, 0) + 1) / 2)
            plt.axis("off")
        plt.show()
    except KeyboardInterrupt:
        pass

