import json
import os
import requests
import tiktoken
import torch
import pandas as pd
import zipfile
import torchvision
import torchvision.transforms.functional
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

class TextImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        assert 'caption' in df.columns
        assert 'image_path' in df.columns
        self.df = df
        self.tokenizer = tiktoken.get_encoding("gpt2")
    
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
        return TextImageDataset.load(os.path.expanduser("~/Downloads/COCO/annotations_trainval2017/annotations/captions_train2017.json"), os.path.expanduser("~/Downloads/COCO/train2017"))
    
    @staticmethod
    def load_valid():
        return TextImageDataset.load(os.path.expanduser("~/Downloads/COCO/annotations_trainval2017/annotations/captions_val2017.json"), os.path.expanduser("~/Downloads/COCO/val2017"))
    
    @staticmethod
    def download(valid: bool = False):
        if valid:
            imgs = "http://images.cocodataset.org/zips/val2017.zip"
        else:
            imgs = "http://images.cocodataset.org/zips/train2017.zip"
        annotations = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        coco_path = os.path.expanduser("~/Downloads/COCO")
        os.makedirs(coco_path, exist_ok=True)

        # Downloading imgs2017.zip
        imgs_path = f"{coco_path}/train2017.zip" if not valid else f"{coco_path}/val2017.zip"
        if not os.path.exists(imgs_path):
            val_imgs_stream = requests.get(imgs, stream=True)
            val_size = int(val_imgs_stream.headers.get("Content-Length", 0))
            with open(imgs_path, "wb") as f:
                with tqdm(
                    total=val_size, 
                    unit="B", 
                    unit_scale=True, 
                    desc="Downloading images") as bar:
                    for chunk in val_imgs_stream.iter_content(chunk_size=4096):
                        f.write(chunk)
                        bar.update(len(chunk))
        
        # Downloading annotations_trainval2017.zip
        if not os.path.exists(f"{coco_path}/annotations_trainval2017.zip"):
            annotations_stream = requests.get(annotations, stream=True)
            annotations_size = int(annotations_stream.headers.get("Content-Length", 0))
            with open(f"{coco_path}/annotations_trainval2017.zip", "wb") as f:
                with tqdm(
                    total=annotations_size, 
                    unit="B", 
                    unit_scale=True, 
                    desc="Downloading annotations") as bar:
                    for chunk in annotations_stream.iter_content(chunk_size=4096):
                        f.write(chunk)
                        bar.update(len(chunk))
        
        # Extracting val2017.zip
        imgs_path = imgs_path[:-4] # remove .zip
        if not os.path.exists(imgs_path):
            with zipfile.ZipFile(f"{imgs_path}.zip", "r") as zip_ref:
                zip_ref.extractall(coco_path)

        # Extracting annotations_trainval2017.zip
        os.makedirs(f"{coco_path}/annotations_trainval2017", exist_ok=True)
        if not os.path.exists(f"{coco_path}/annotations_trainval2017/annotations"):
            with zipfile.ZipFile(f"{coco_path}/annotations_trainval2017.zip", "r") as zip_ref:
                zip_ref.extractall(f"{coco_path}/annotations_trainval2017")

        if valid:
            return TextImageDataset.load_valid()
        else:
            return TextImageDataset.load_train()

    def __len__(self):  
        return len(self.df)

    def _load_image(self, image_path: str):
        image = torchvision.io.decode_image(torchvision.io.read_file(image_path))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
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
        return images, captions, lengths, plaintext_caption

if __name__ == "__main__":
    dataset = TextImageDataset.load_train()
    l = 3
    for i, item in enumerate(dataset):
        plt.subplot(l, l, i + 1)
        plt.imshow((item[0].permute(1, 2, 0) + 1) / 2)
        plt.title(f"{dataset.tokenizer.decode(item[1].cpu().numpy())}\n{dataset.tokenizer.decode([item[2].item()])}")
        plt.axis("off")
        if i == l * l - 1:
            break
    plt.show()