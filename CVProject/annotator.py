import os
from typing import Optional
import torch
from torch import nn
import tiktoken
from tqdm.auto import tqdm

from CVProject.dataset import TextImageDataset
from CVProject.vit import ViT

class Annotator(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 32
        self.dropout = 0.2

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.max_token_value + 1,
            embedding_dim=self.embedding_dim,
        )

        self.encoder = ViT(
            conv_depth=3,
            embedding_dim=self.embedding_dim,
            filter_size=3,
            dropout=self.dropout,
            transformer_depth=1,
            transformer_width=self.embedding_dim,
            transformer_heads=4)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=4,
                dim_feedforward=self.embedding_dim,
                dropout=self.dropout,
                batch_first=True),
                num_layers=1,
                norm=nn.LayerNorm(self.embedding_dim)
            )
        
        self.deembedding = nn.Linear(self.embedding_dim, self.tokenizer.max_token_value + 1)
        self.to(self.device)

    def forward(self, generated_caption, masks, images):
        image_embeddings = self.encoder(images)
        caption_embeddings = self.embedding(generated_caption)
        output = self.decoder(
            tgt=caption_embeddings, 
            memory=image_embeddings, 
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(caption_embeddings.shape[1], device=self.device),
            tgt_is_causal=True,)
        last_output = output[:, -1, :]
        return self.deembedding(last_output)

    def fit(self, train: TextImageDataset, valid: TextImageDataset, epochs: int, _only_one_batch: bool = False):
        optim = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        bs = 32
        train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=bs, collate_fn=TextImageDataset.collate_fn)
        valid_loader= torch.utils.data.DataLoader(dataset=valid, batch_size=bs, collate_fn=TextImageDataset.collate_fn)

        for epoch in range(epochs):
            batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (train)")
            self.train()
            avg_loss = 0
            correct_predictions = 0
            total_predictions = 0
            for i, (images, captions, masks, next_tokens) in enumerate(batches):
                images = images.to(self.device)
                captions = captions.to(self.device)
                masks = masks.to(self.device)
                next_tokens = next_tokens.to(self.device)
                optim.zero_grad()
                generated_tokens = self(captions, masks, images)
                loss: torch.Tensor = loss_fn(generated_tokens, next_tokens)
                loss.backward()
                optim.step()
                avg_loss += loss.item()
                correct_predictions += (generated_tokens.argmax(dim=-1) == next_tokens).sum().item()
                total_predictions += next_tokens.numel()
                batches.set_postfix(loss=avg_loss / (i + 1), accuracy=f"{correct_predictions / total_predictions:0.2%}")
                if _only_one_batch:
                    break
            batches = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} (valid)")
            self.eval()
            avg_loss = 0
            correct_predictions = 0
            total_predictions = 0
            with torch.no_grad():
                for i, (images, captions, masks, next_tokens) in enumerate(batches):
                    images = images.to(self.device)
                    captions = captions.to(self.device)
                    masks = masks.to(self.device)
                    next_tokens = next_tokens.to(self.device)
                    generated_tokens = self(captions, masks, images)
                    loss: torch.Tensor = loss_fn(generated_tokens, next_tokens)
                    avg_loss += loss.item()
                    correct_predictions += (generated_tokens.argmax(dim=-1) == next_tokens).sum().item()
                    total_predictions += next_tokens.numel()
                    batches.set_postfix(loss=avg_loss / (i + 1), accuracy=f"{correct_predictions / total_predictions:0.2%}")
                    if _only_one_batch:
                        break
            avg_loss /= i + 1
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    @staticmethod
    def load(path: str):
        annotator = Annotator()
        annotator.load_state_dict(torch.load(path, map_location=annotator.device, weights_only=True))
        return annotator
    
    def annotate(self, image, max_length: Optional[int] = 20):
        self.eval()
        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0)
            caption = torch.tensor([self.tokenizer.max_token_value], dtype=torch.long).unsqueeze(0).to(self.device)
            generated_caption = []
            not_finished = False
            while True:
                masks = torch.ones((1, len(generated_caption) + 1), dtype=torch.bool).to(self.device)
                caption = caption.to(self.device)
                generated_tokens = self(caption, masks, image)
                generated_token = torch.multinomial(torch.softmax(generated_tokens, dim=-1), 1).squeeze(1)
                
                if generated_token == self.tokenizer.max_token_value:
                    break
                generated_caption.append(generated_token)
                caption = torch.cat([caption, generated_token.unsqueeze(0)], dim=1)
                if max_length is not None and len(generated_caption) >= max_length:
                    not_finished = True
                    break
            append = "..." if not_finished else ""
            return self.tokenizer.decode(generated_caption) + append
        
if __name__ == "__main__":
    train = TextImageDataset.load_train()
    valid = TextImageDataset.load_valid()
    annotator = Annotator()
    if os.path.exists("data/annotator.pt"):
        annotator = Annotator.load("data/annotator.pt")
    else:
        try:
            annotator.fit(train, valid, epochs=10)
        except KeyboardInterrupt:
            pass
    annotator.save("data/annotator.pt")
    import matplotlib.pyplot as plt
    data = next(iter(valid))
    print(annotator.tokenizer.decode(data[1].numpy()))
    img, _, _ = data
    cap = annotator.annotate(img)
    plt.imshow((img.permute(1, 2, 0) + 1) / 2)
    plt.axis("off")
    plt.title(cap)
    plt.show()
    
    
