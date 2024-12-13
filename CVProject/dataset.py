import json
import os
import tiktoken
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision
import torchvision.transforms.functional
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

class TextImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        assert 'caption' in df.columns
        assert 'image_path' in df.columns
        self.df = df
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.target_size = 224
    
    @staticmethod
    def load(captions_path: str, images_path: str):
        img_name_padding = 12
        
        with open(captions_path, "r") as f:
            captions_data = json.load(f)["annotations"]
        df_list = []
        for caption_data in tqdm(captions_data, desc="Loading dataset"):
            image_id = caption_data["image_id"]
            caption = caption_data["caption"]
            image_name = f"{str(image_id).zfill(img_name_padding)}.jpg"
            image_path = os.path.join(images_path, image_name)
            df_list.append({"caption": caption, "image_path": image_path})
        df = pd.DataFrame.from_dict(df_list)
        return TextImageDataset(df)
    
    @staticmethod
    def load_train():
        return TextImageDataset.load("/home/ettore/Downloads/COCO/annotations_trainval2017/annotations/captions_train2017.json", "/home/ettore/Downloads/COCO/train2017")
    
    @staticmethod
    def load_valid():
        return TextImageDataset.load("/home/ettore/Downloads/COCO/annotations_trainval2017/annotations/captions_val2017.json", "/home/ettore/Downloads/COCO/val2017")
    
    def __len__(self):  
        return len(self.df)

    def _load_image(self, image_path: str):
        image = torchvision.io.decode_image(torchvision.io.read_file(image_path))
        # image = torchvision.transforms.functional.resize(
        #     image, 
        #     (self.target_size, self.target_size), 
        #     torchvision.transforms.InterpolationMode.BILINEAR)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        # image = (image.to(torch.float) - 127.5) / 127.5
        return image
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = row["caption"]
        plaintext_caption = caption
        image_path = row["image_path"]
        image = self._load_image(image_path)
        caption = [self.tokenizer.max_token_value] + self.tokenizer.encode(caption) + [self.tokenizer.max_token_value]
        caption = torch.tensor(caption, dtype=torch.long)
        return image, caption, plaintext_caption
    
    @staticmethod
    def collate_fn(batch: list):
        images, captions, plaintext_caption = zip(*batch)
        lengths = torch.tensor([len(c) for c in captions])
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
        # images = torch.stack(images)
        return images, captions, lengths, plaintext_caption

if __name__ == "__main__":
    dataset = TextImageDataset.load_train()
    l = 3
    for i, item in enumerate(dataset):
        plt.subplot(l, l, i + 1)
        plt.imshow((item[0].permute(1, 2, 0) + 1) / 2)
        plt.title(f"{dataset.tokenizer.decode(item[1].numpy())}\n{dataset.tokenizer.decode([item[2].item()])}")
        plt.axis("off")
        if i == l * l - 1:
            break
    plt.show()