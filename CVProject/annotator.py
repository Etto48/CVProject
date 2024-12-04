import os
from typing import Literal, Optional
import torch
from torch import nn
import tiktoken
from tqdm.auto import tqdm

from CVProject.dataset import TextImageDataset
from CVProject.lambda_layer import LambdaLayer
from CVProject.pos_enc import PositionalEncoding
from CVProject.vit import ViT


class Annotator(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 128
        self.heads = 8
        self.dropout = 0.1

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.tokenizer.max_token_value + 1,
                embedding_dim=self.embedding_dim,
            ),
            nn.LayerNorm(self.embedding_dim),
            PositionalEncoding()
        )
        self.encoder = ViT(
            embedding_dim=self.embedding_dim,
            patch_size=32,
            dropout=self.dropout,
            transformer_depth=1,
            transformer_width=self.embedding_dim * self.heads,
            transformer_heads=self.heads)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=self.heads,
                dim_feedforward=self.embedding_dim * self.heads,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu"),
            num_layers=1,
            norm=nn.LayerNorm(self.embedding_dim)
        )

        self.deembedding = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.tokenizer.max_token_value + 1)
        )
        self.to(self.device)

    def forward(self, generated_caption, lengths, images):
        image_embeddings: torch.Tensor = self.encoder(images)
        caption_embeddings: torch.Tensor = self.embedding(generated_caption)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            caption_embeddings.shape[1], device=self.device)
        output = self.decoder(
            tgt=caption_embeddings,
            memory=image_embeddings,
            tgt_mask=tgt_mask,
            tgt_is_causal=True)
        last_output = output[torch.arange(output.shape[0]), lengths - 1]
        image_class_embedding = image_embeddings[:, 0]
        last_output = torch.cat([last_output, image_class_embedding], dim=1)
        return self.deembedding(last_output)

    def fit(self, train: TextImageDataset, valid: TextImageDataset, epochs: int):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        history = {"train": [], "valid": []}
        loss_fn = nn.CrossEntropyLoss()
        bs = 32
        train_loader = torch.utils.data.DataLoader(
            dataset=train, batch_size=bs, collate_fn=TextImageDataset.collate_fn)
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid, batch_size=bs, collate_fn=TextImageDataset.collate_fn)

        best_loss = float("inf")
        best_model = None
        patience = 4
        epochs_without_improvement = 0
        threshold = 0.01

        for epoch in range(epochs):
            batches = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs} (train)")
            self.train()
            avg_loss = 0
            correct_predictions = 0
            total_predictions = 0
            for i, (images, captions, lengths, next_tokens, _) in enumerate(batches):
                images = images.to(self.device)
                captions = captions.to(self.device)
                lengths = lengths.to(self.device)
                next_tokens = next_tokens.to(self.device)
                optim.zero_grad()
                generated_tokens = self(captions, lengths, images)
                loss: torch.Tensor = loss_fn(generated_tokens, next_tokens)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step()
                avg_loss += loss.item()
                correct_predictions += (generated_tokens.argmax(dim=-1)
                                        == next_tokens).sum().item()
                total_predictions += next_tokens.numel()
                batches.set_postfix(
                    loss=avg_loss / (i + 1), accuracy=f"{correct_predictions / total_predictions:0.2%}")
            avg_loss /= i + 1
            history["train"].append(avg_loss)
            batches = tqdm(
                valid_loader,
                desc=f"Epoch {epoch + 1}/{epochs} (valid)")
            self.eval()
            avg_loss = 0
            correct_predictions = 0
            total_predictions = 0
            with torch.no_grad():
                for i, (images, captions, lengths, next_tokens, _) in enumerate(batches):
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
            avg_loss /= i + 1
            history["valid"].append(avg_loss)
            if avg_loss < best_loss - threshold:
                best_loss = avg_loss
                best_model = self.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        self.load_state_dict(best_model)
        return history

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: str):
        annotator = Annotator()
        annotator.load_state_dict(torch.load(
            path, map_location=annotator.device, weights_only=True))
        return annotator

    def annotate(self, images, max_length: Optional[int] = 15, mode: Literal["greedy", "sample"] = "sample", top_k: Optional[int] = None):
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
            probabilities = torch.tensor(
                [1.0], dtype=torch.float32, device=self.device).repeat(images.shape[0])
            while True:
                generated_tokens_distribution = self(captions, lengths, images)
                match mode:
                    case "greedy":
                        generated_tokens = generated_tokens_distribution.argmax(
                            dim=-1)
                    case "sample":
                        if top_k is not None:
                            top_k_values, top_k_indices = torch.topk(
                                generated_tokens_distribution, top_k, dim=-1)
                            top_k_probabilities = torch.softmax(
                                top_k_values, dim=-1)
                            generated_tokens = torch.multinomial(
                                top_k_probabilities, 1)\
                                .squeeze(1)
                            generated_tokens = top_k_indices[
                                torch.arange(generated_tokens.shape[0]), generated_tokens]
                        else:
                            generated_tokens = torch.multinomial(
                                torch.softmax(generated_tokens_distribution, dim=-1), 1)\
                                .squeeze(1)
                        
                probabilities *= torch.softmax(
                    generated_tokens_distribution,
                    dim=-1)[torch.arange(generated_tokens.shape[0]), generated_tokens]

                finished |= \
                    (generated_tokens == self.tokenizer.max_token_value)
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
            return outputs, probabilities
        else:
            assert len(outputs) == 1
            return outputs[0], probabilities[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train = TextImageDataset.load_train()
    valid = TextImageDataset.load_valid()
    annotator = Annotator()
    if os.path.exists("data/annotator.pt"):
        annotator = Annotator.load("data/annotator.pt")
    else:
        try:
            history = annotator.fit(train, valid, epochs=100)
        except KeyboardInterrupt:
            history = None
            pass
    annotator.save("data/annotator.pt")
    if history is not None:
        plt.plot(history["train"], label="Train")
        plt.plot(history["valid"], label="Valid")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    display_demo = input("Do you want to display a demo? [y/N]: ")
    display_demo = display_demo.lower() == "y"
    if display_demo:
        l = 3
        images = []
        og_captions = []
        for i, data in enumerate(valid):
            images.append(data[0])
            og_captions.append(data[3])
            if i == l * l - 1:
                break

        images = torch.stack(images)
        captions, probabilities = annotator.annotate(images, mode="sample", top_k=5)
        for i in range(l * l):
            plt.subplot(l, l, i + 1)
            plt.imshow((images[i].permute(1, 2, 0) + 1) / 2)
            plt.title(captions[i] + "\n" + f"{probabilities[i]:.8%}")
            plt.axis("off")
        plt.show()
