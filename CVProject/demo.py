
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
    captions_beam, beam_prob = annotator.annotate(images, mode="beam", top_k=10)
    captions_greedy, greedy_prob = annotator.annotate(images, mode="greedy")
    captions_sample, sample_prob = annotator.annotate(images, mode="sample", top_k=10)
    plt.figure(figsize=(25, 15))
    for i in range(l * l):
        plt.subplot(l, l, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(f"Beam: {captions_beam[i]} ({beam_prob[i]:.3%})\n"
                  f"Greedy: {captions_greedy[i]} ({greedy_prob[i]:.3%})\n"
                  f"Sample: {captions_sample[i]} ({sample_prob[i]:.3%})")
        plt.axis("off")
    plt.show()

    