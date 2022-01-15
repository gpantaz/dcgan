import os
import torch
import yaml
import argparse
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models.dcgan import Generator, Discriminator, weights_init
from utils.utils import get_noise, plot_images


def training_step(models, optimizers, criterion, batch, device):
    """Step during training."""
    generator, discriminator = models[0], models[1]
    gen_opt, disc_opt = optimizers[0], optimizers[1]

    batch = batch.to(device)

    # Update discriminator
    disc_opt.zero_grad()
    fake_noise = get_noise(
        n_samples=len(batch), z_dim=generator.noise_dim, device=device
    )
    fake = generator(fake_noise)
    disc_fake_pred = discriminator(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = discriminator(batch)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    disc_loss.backward(retain_graph=True)
    disc_opt.step()

    # Update generator
    gen_opt.zero_grad()
    fake_noise = get_noise(
        n_samples=len(batch), z_dim=generator.noise_dim, device=device
    )
    fake = generator(fake_noise)
    disc_fake_pred = discriminator(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    gen_loss.backward()
    gen_opt.step()
    return gen_loss, disc_loss, fake


def main(args):
    """Main function."""

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    if config["visualizations"]["visualize"]:
        os.makedirs(config["visualizations"]["output_dir"], exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = MNIST(".", download=True, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=config["trainer"]["batch_size"], shuffle=True
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    display_step = 500

    train_on_gpu = torch.cuda.is_available()
    device = "cuda" if train_on_gpu else "cpu"

    generator = Generator(config=config).to(device)
    gen_opt = torch.optim.Adam(
        generator.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta_1"], config["optimizer"]["beta_2"]),
    )
    discriminator = Discriminator(config=config).to(device)
    disc_opt = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta_1"], config["optimizer"]["beta_2"]),
    )

    cur_step = 0
    for epoch in range(config["trainer"]["epochs"]):
        avg_gen_loss = 0
        avg_disc_loss = 0
        avg_step = 1
        pbar = tqdm(dataloader)
        for batch, _ in pbar:
            gen_loss, disc_loss, fake = training_step(
                models=[generator, discriminator],
                optimizers=[gen_opt, disc_opt],
                criterion=criterion,
                batch=batch,
                device=device,
            )
            if (
                config["visualizations"]["visualize"]
                and cur_step % display_step == 0
                and cur_step > 0
            ):
                plot_images(
                    fake,
                    name=os.path.join(
                        config["visualizations"]["output_dir"], "fake_{}.png"
                    ).format(cur_step),
                )
                plot_images(
                    batch,
                    name=os.path.join(
                        config["visualizations"]["output_dir"], "real_{}.png"
                    ).format(cur_step),
                )
            avg_gen_loss += gen_loss.item()
            avg_disc_loss += disc_loss.item()
            msg = "Epoch: {} Running loss: Gen {}, Disc {}".format(
                epoch + 1,
                round(avg_gen_loss / avg_step, 3),
                round(avg_disc_loss / avg_step, 3),
            )
            pbar.set_description(msg)
            cur_step += 1
            avg_step += 1


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(prog="PROG")
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/dcgan.yaml",
        help="Path to configuration file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
