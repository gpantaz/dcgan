import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def get_noise(n_samples, z_dim, device="cpu"):
    """Samples noise vectors."""
    return torch.randn(n_samples, z_dim, device=device)


def plot_images(image_tensor, name, num_images=25):
    """Plots a sample of images."""
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(name)
