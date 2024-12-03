import os
from typing import Literal, Optional
import torch
from torch import nn
import tiktoken
from tqdm.auto import tqdm

from CVProject.dataset import TextImageDataset
from CVProject.pos_enc import PositionalEncoding
from CVProject.vit import ViT


class Annotator(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 64
        self.heads = 8
        self.dropout = 0.2

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.max_token_value + 1,
            embedding_dim=self.embedding_dim,
        )
        self.pos_enc = PositionalEncoding(
            self.embedding_dim, device=self.device)

        self.encoder = ViT(
            conv_depth=3,
            embedding_dim=self.embedding_dim,
            filter_size=3,
            dropout=self.dropout,
            transformer_depth=1,
            transformer_width=self.embedding_dim,
            transformer_heads=self.heads)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=self.heads,
                dim_feedforward=self.embedding_dim,
                dropout=self.dropout,
                batch_first=True),
            num_layers=1,
            norm=nn.LayerNorm(self.embedding_dim)
        )

        self.deembedding = nn.Linear(
            self.embedding_dim, self.tokenizer.max_token_value + 1)
        self.to(self.device)

    def forward(self, generated_caption, lengths, images):
        image_embeddings = self.encoder(images)
        caption_embeddings = self.embedding(generated_caption)
        caption_embeddings = self.pos_enc(caption_embeddings)
        output = self.decoder(
            tgt=caption_embeddings,
            memory=image_embeddings,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                caption_embeddings.shape[1], device=self.device),
            tgt_is_causal=True,)
        last_output = output[torch.arange(output.shape[0]), lengths - 1]
        return self.deembedding(last_output)

    def fit(self, train: TextImageDataset, valid: TextImageDataset, epochs: int, _only_one_batch: bool = False):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        bs = 32
        train_loader = torch.utils.data.DataLoader(
            dataset=train, batch_size=bs, collate_fn=TextImageDataset.collate_fn)
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid, batch_size=bs, collate_fn=TextImageDataset.collate_fn)

        for epoch in range(epochs):
            batches = tqdm(
                train_loader, 
                desc=f"Epoch {epoch + 1}/{epochs} (train)")
            self.train()
            avg_loss = 0
            correct_predictions = 0
            total_predictions = 0
            for i, (images, captions, lengths, next_tokens) in enumerate(batches):
                images = images.to(self.device)
                captions = captions.to(self.device)
                lengths = lengths.to(self.device)
                next_tokens = next_tokens.to(self.device)
                optim.zero_grad()
                generated_tokens = self(captions, lengths, images)
                loss: torch.Tensor = loss_fn(generated_tokens, next_tokens)
                loss.backward()
                optim.step()
                avg_loss += loss.item()
                correct_predictions += (generated_tokens.argmax(dim=-1)
                                        == next_tokens).sum().item()
                total_predictions += next_tokens.numel()
                batches.set_postfix(
                    loss=avg_loss / (i + 1), accuracy=f"{correct_predictions / total_predictions:0.2%}")
                if _only_one_batch:
                    break
            batches = tqdm(
                valid_loader, 
                desc=f"Epoch {epoch + 1}/{epochs} (valid)")
            self.eval()
            avg_loss = 0
            correct_predictions = 0
            total_predictions = 0
            with torch.no_grad():
                for i, (images, captions, lengths, next_tokens) in enumerate(batches):
                    images = images.to(self.device)
                    captions = captions.to(self.device)
                    lengths = lengths.to(self.device)
                    next_tokens = next_tokens.to(self.device)
                    generated_tokens = self(captions, lengths, images)
                    loss: torch.Tensor = loss_fn(generated_tokens, next_tokens)
                    avg_loss += loss.item()
                    correct_predictions += (generated_tokens.argmax(dim=-1)
                                            == next_tokens).sum().item()
                    total_predictions += next_tokens.numel()
                    batches.set_postfix(
                        loss=avg_loss / (i + 1), accuracy=f"{correct_predictions / total_predictions:0.2%}")
                    if _only_one_batch:
                        break
            avg_loss /= i + 1

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: str):
        annotator = Annotator()
        annotator.load_state_dict(torch.load(
            path, map_location=annotator.device, weights_only=True))
        return annotator

    def annotate(self, images, max_length: Optional[int] = 10, mode: Literal["greedy", "sample"] = "sample"):
        self.eval()
        batched = images.dim() == 4
        with torch.no_grad():
            images = images.to(self.device)
            images = images.unsqueeze(0) if images.dim() == 3 else images

            captions = torch.tensor([self.tokenizer.max_token_value], dtype=torch.long, device=self.device)\
                .unsqueeze(0)\
                .repeat(images.shape[0], 1)
            finished = torch.tensor(
                [False], dtype=torch.bool, device=self.device).repeat(images.shape[0])
            lengths = torch.tensor(
                [1], dtype=torch.long, device=self.device).repeat(images.shape[0])
            while True:
                generated_tokens = self(captions, lengths, images)
                match mode:
                    case "greedy":
                        generated_tokens = generated_tokens.argmax(dim=-1)
                    case "sample":
                        generated_token = torch.multinomial(
                            torch.softmax(generated_tokens, dim=-1),
                            1)\
                            .squeeze(1)

                finished |= (generated_tokens ==
                             self.tokenizer.max_token_value)
                if finished.all():
                    break
                lengths += ~finished

                captions = torch.cat(
                    [captions, generated_tokens.unsqueeze(1)], dim=1)
                if max_length is not None and captions.shape[1] >= max_length:
                    break
        outputs = []
        for i in range(captions.shape[0]):
            caption = captions[i, 1:lengths[i]].cpu().numpy()
            output = self.tokenizer.decode(caption)
            if not finished[i]:
                output += "..."
            outputs.append(output)
        if batched:
            return outputs
        else:
            assert len(outputs) == 1
            return outputs[0]


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
    l = 3
    images = []
    for i, data in enumerate(valid):
        images.append(data[0])
        if i == l * l - 1:
            break
    images = torch.stack(images)
    captions = annotator.annotate(images)
    for i in range(l * l):
        plt.subplot(l, l, i + 1)
        plt.imshow((images[i].permute(1, 2, 0) + 1) / 2)
        plt.title(captions[i])
        plt.axis("off")
    plt.show()
