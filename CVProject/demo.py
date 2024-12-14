
from matplotlib import pyplot as plt
import torch
from CVProject.annotator import Annotator
from CVProject.dataset import TextImageDataset

if __name__ == "__main__":
    try:
        print("Loading annotator...")
        annotator = Annotator.load("data/annotator.pt")
    except FileNotFoundError:
        print("Annotator pretrained weights not found at \"data/annotator.pt\"")
        exit(1)
    valid = TextImageDataset.load_valid()
    l = 3
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid, batch_size=l * l, collate_fn=TextImageDataset.collate_fn,
        sampler=torch.utils.data.RandomSampler(valid, replacement=True, num_samples=l * l))
    images, _, _, _ = next(iter(valid_loader))
    captions, probabilities = annotator.annotate(images, mode="beam", top_k=5)
    for i in range(l * l):
        plt.subplot(l, l, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(captions[i] + "\n" + f"{probabilities[i]:.4%}")
        plt.axis("off")
    plt.show()

    