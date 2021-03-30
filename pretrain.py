from utils.transforms import get_no_aug_transform
from utils.helpers import unnormalize
import torch
from torch.optim import AdamW
import torchvision.utils as vutils

from utils.loss import ContentLoss
from utils.datasets import get_dataloader
from models.generator import Generator
from tqdm import tqdm


def train():
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    batch_size = 16
    image_size = 256
    learning_rate = 1e-3
    beta1, beta2 = (.5, .99)
    weight_decay = 1e-3
    epochs = 10

    # Dataloaders
    real_dataloader = get_dataloader(
        "./datasets/real_images/flickr_nuneaton/", size=image_size, bs=batch_size, trfs=get_no_aug_transform())

    # Lists to keep track of progress
    G_losses = []
    iters = 0

    tracked_images = next(iter(real_dataloader)).to(device)
    vutils.save_image(unnormalize(tracked_images),
                      "images/org.png", padding=2, normalize=True)

    # Models
    netG = Generator().to(device)

    scaler = torch.cuda.amp.GradScaler()

    optimizerG = AdamW(netG.parameters(), lr=learning_rate,
                       betas=(beta1, beta2), weight_decay=weight_decay)

    # Loss functions
    content_loss = ContentLoss().to(device)

    print("Starting Training Loop...")
    # For each epoch.
    for epoch in range(epochs):
        # For each batch in the dataloader.
        for i, real_data, in enumerate(tqdm(real_dataloader, desc=f"Training epoch {epoch}")):

            ############################
            # (1) Pre-train G
            ###########################

            # Reset Discriminator gradient.
            netG.zero_grad()

            # Format batch.
            real_data = real_data.to(device)

            with torch.cuda.amp.autocast():
                # Generate image
                generated_data = netG(real_data)

                # Calculate discriminator loss on all batches.
                errG = content_loss(generated_data, real_data)

            # Calculate gradients for G
            scaler.scale(errG).backward()

            # Update G
            scaler.step(optimizerG)

            scaler.update()

            # ---------------------------------------------------------------------------------------- #

            # Save Losses for plotting later
            G_losses.append(errG.item())

            # Check how the generator is doing by saving G's output on tracked_images
            if iters % 200 == 0:
                with torch.no_grad():
                    fake = netG(tracked_images).detach().cpu()
                vutils.save_image(unnormalize(
                    fake), f"images/{epoch}_{i}.png", padding=2)
                torch.save(netG, f"checkpoints/pretrained_netG_e{epoch}_i{iters}_l{errG.item()}.pth")

            iters += 1

    torch.save(netG.state_dict(), f"checkpoints/pretrained_netG_e{epoch}_i{iters}_l{errG.item()}.pth")

if __name__ == "__main__":
    train()
