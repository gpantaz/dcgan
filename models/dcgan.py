import torch
from torch import nn
from torchvision.datasets import MNIST


class Generator(nn.Module):
    """Generator Class."""

    def __init__(self, img_channels=1, config=None):
        super().__init__()

        self.config = config["generator"]
        self.img_channels = img_channels
        self.noise_dim = self.config["noise_dim"]
        self.config["out_channels"].append(img_channels)
        self.gen = self.create_gen_blocks()

    def create_gen_blocks(self):
        """Creates generator blocks."""
        gen_blocks = []
        in_channels = self.noise_dim
        for idx in range(self.config["layers"]):
            out_channels = self.config["out_channels"][idx]
            gen_blocks.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.config["kernel_size"][idx],
                    stride=self.config["stride"][idx],
                ),
            )
            if self.config["batch_normalization"][idx]:
                gen_blocks.append(nn.BatchNorm2d(num_features=out_channels))

            activ = self._make_activ(self.config["activation"][idx])
            gen_blocks.append(activ)
            in_channels = out_channels

        return nn.Sequential(*gen_blocks)

    def unsqueeze_noise(self, noise):
        """Helper function for forward pass."""
        return noise.view(len(noise), self.noise_dim, 1, 1)

    def forward(self, noise):
        """Perform forward pass."""
        noise_expand = self.unsqueeze_noise(noise)
        return self.gen(noise_expand)

    def _make_activ(self, activ="relu"):
        if activ == "relu":
            activation = nn.ReLU()
        elif activ == "tanh":
            activation = nn.Tanh()
        elif activ == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError(f"Unsupported activ function {activ}")
        return activation


class Discriminator(nn.Module):
    """Discriminator Class."""

    def __init__(self, img_channels=1, config=None):
        super().__init__()

        self.config = config["discriminator"]
        self.img_channels = img_channels
        self.config["out_channels"].append(1)
        self.disc = self.create_disc_blocks()

    def create_disc_blocks(self):
        """Creates discriminator blocks."""
        disc_blocks = []
        in_channels = self.img_channels
        for idx in range(self.config["layers"]):
            out_channels = self.config["out_channels"][idx]
            disc_blocks.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.config["kernel_size"][idx],
                    stride=self.config["stride"][idx],
                ),
            )
            if self.config["batch_normalization"][idx]:
                disc_blocks.append(nn.BatchNorm2d(num_features=out_channels))

            if idx < self.config["layers"] - 1:
                activ = self._make_activ(self.config["activation"][idx])
                disc_blocks.append(activ)

            in_channels = out_channels

        return nn.Sequential(*disc_blocks)

    def forward(self, image):
        """
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

    def _make_activ(self, activ="relu"):
        if activ == "relu":
            activation = nn.ReLU()
        elif activ == "tanh":
            activation = nn.Tanh()
        elif activ == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError(f"Unsupported activ function {activ}")
        return activation


def get_noise(n_samples, z_dim, device="cpu"):
    """Sample noise vectors.

    Args:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """
    return torch.randn(n_samples, z_dim, device=device)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
