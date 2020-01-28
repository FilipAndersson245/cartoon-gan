import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from utils.loss import ContentLoss, AdversialLoss
from utils.transforms import get_default_transforms
from utils.datasets import get_dataloader
from models.discriminator import Discriminator
from models.generator import Generator


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    batch_size = 9
    image_size = 256
    learning_rate = 1e-3
    beta1, beta2 = (.5, .99)
    weight_decay = 1e-3
    epochs = 10

    # Models
    netD = Discriminator().to(device)
    netG = Generator().to(device)

    optimizerD = AdamW(netD.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    optimizerG = AdamW(netG.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    # Labels
    cartoon_labels = torch.ones (batch_size, 1, image_size // 4, image_size // 4).to(device)
    fake_labels    = torch.zeros(batch_size, 1, image_size // 4, image_size // 4).to(device)

    # Loss functions
    content_loss = ContentLoss(device)
    adv_loss     = AdversialLoss(cartoon_labels, fake_labels)
    BCE_loss     = nn.BCELoss().to(device)

    # Dataloaders
    real_dataloader    = get_dataloader("./datasets/real_images",           size = image_size, bs = batch_size)
    cartoon_dataloader = get_dataloader("./datasets/cartoon_images",        size = image_size, bs = batch_size)
    edge_dataloader    = get_dataloader("./datasets/cartoon_images_smooth", size = image_size, bs = batch_size)

    # --------------------------------------------------------------------------------------------- #
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    tracked_images = next(iter(real_dataloader))[0].to(device)

    print("Starting Training Loop...")
    # For each epoch.
    for epoch in range(epochs):
        # For each batch in the dataloader.
        for i, ((cartoon_data, _), (edge_data, _), (real_data, _)) in enumerate(zip(cartoon_dataloader, edge_dataloader, real_dataloader )):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            
            # Reset Discriminator gradient.
            netD.zero_grad()

            # Format batch.
            cartoon_data   = cartoon_data.to(device)
            edge_data      = edge_data.to(device)
            real_data      = real_data.to(device)

            # Generate image
            generated_data = netG(real_data)

            # Forward pass all batches through D.
            cartoon_pred   = netD(cartoon_data)      #.view(-1)
            edge_pred      = netD(edge_data)         #.view(-1)
            generated_pred = netD(generated_data)    #.view(-1)

            print(generated_data.is_cuda, real_data.is_cuda)

            # Calculate discriminator loss on all batches.
            errD = adv_loss(cartoon_pred, generated_pred, edge_pred)
            
            # Calculate gradients for D in backward pass
            errD.backward()
            D_x = cartoon_pred.mean().item() # Should be close to 1

            # Update D
            optimizerD.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            
            # Reset Generator gradient.
            netG.zero_grad()
            
            # Since we just updated D, perform another forward pass of all-fake batch through D
            generated_pred = netD(generated_data) #.view(-1)

            # Calculate G's loss based on this output
            print(generated_data.is_cuda, real_data.is_cuda)
            print("generated_pred:", generated_pred.is_cuda, "cartoon_labels:", cartoon_labels.is_cuda)
            errG = BCE_loss(generated_pred, cartoon_labels) + content_loss(generated_data, real_data)

            # Calculate gradients for G
            errG.backward()

            D_G_z2 = generated_pred.mean().item() # Should be close to 1
            
            # Update G
            optimizerG.step()
            
            # ---------------------------------------------------------------------------------------- #

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(real_dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on tracked_images
            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(tracked_images).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1



if __name__ == "__main__":
    train()
