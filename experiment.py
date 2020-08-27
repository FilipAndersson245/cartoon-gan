import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
import torchvision.utils as vutils
from torchvision.models import vgg16

# from utils.loss import ContentLoss, AdversialLoss
from utils.transforms import get_photo_train_loader, get_cartoon_train_loader
from utils.transforms import get_pair_transforms
from torch.utils.tensorboard import SummaryWriter
from models.discriminator import Discriminator
from models.generator import Generator

from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import os
import itertools
import gc
from tqdm import tqdm

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
run_ID = datetime.now().strftime("%Y.%m.%d %H.%M")
batch_size = 11
image_size = 224
learning_rate = 1.5e-4
beta1, beta2 = (.5, .99)
weight_decay = 1e-4
epochs = 100
n_critic_iters = 5
log_interval = 5

# Labels
cartoon_labels = torch.ones(
    batch_size, 1, image_size // 4, image_size // 4).to(device)
fake_labels = torch.zeros(batch_size, 1, image_size //
                          4, image_size // 4).to(device)


def inv_normalize(img):
    # Adding 0.1 to all normalization values since the model is trained (erroneously) without correct de-normalization
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    img = img.clamp(0, 1)
    return img


class AdversialLoss(nn.Module):
    def __init__(self, cartoon_labels, fake_labels, device="cpu"):
        super(AdversialLoss, self).__init__()
        self.cartoon_labels = cartoon_labels
        self.fake_labels = fake_labels
        self.base_loss = nn.BCEWithLogitsLoss().to(device)

    def forward(self, cartoon, generated_cartoon):
        D_cartoon_loss = self.base_loss(cartoon, self.cartoon_labels)
        D_generated_fake_loss = self.base_loss(
            generated_cartoon, self.fake_labels)
        return D_cartoon_loss + D_generated_fake_loss


class ContentLoss(nn.Module):
    def __init__(self, omega=10, device="cpu"):
        super(ContentLoss, self).__init__()

        self.base_loss = nn.L1Loss().to(device)
        self.omega = omega

        perception = list(vgg16(pretrained=True).features)[:25]
        self.perception = nn.Sequential(*perception).eval().to(device)

        for param in self.perception.parameters():
            param.requires_grad = False

        gc.collect()

    def forward(self, x1, x2):
        x1 = self.perception(x1)
        x2 = self.perception(x2)

        return self.omega * self.base_loss(x1, x2)


# Loss functions
content_loss = ContentLoss(omega=10, device=device)
adv_loss = AdversialLoss(cartoon_labels, fake_labels)
BCE_loss = nn.BCEWithLogitsLoss().to(device)

photo_dataloader = get_photo_train_loader(
    224, batch_size=batch_size * (n_critic_iters + 1))

cartoon_dataloader = get_cartoon_train_loader(
    224, batch_size=batch_size * n_critic_iters)

# photo_dataloader = get_dataloader(
#     "./datasets/real_images/flickr_31k", size=image_size, bs=batch_size * (n_critic_iters + 1))
# cartoon_dataloader = get_dataloader("./datasets/cartoon_images_smoothed/Studio Ghibli",
#                                     size=image_size, bs=batch_size * n_critic_iters, trfs=get_pair_transforms(image_size))

scaler_D = torch.cuda.amp.GradScaler()
scaler_G = torch.cuda.amp.GradScaler()

# Models
D = Discriminator().to(device)
G = Generator().to(device)

optimizerD = AdamW(D.parameters(), lr=learning_rate,
                   betas=(beta1, beta2), weight_decay=weight_decay)
optimizerG = AdamW(G.parameters(), lr=learning_rate,
                   betas=(beta1, beta2), weight_decay=weight_decay)

schedulerD = CyclicLR(optimizer=optimizerD, base_lr=learning_rate,
                      max_lr=learning_rate*1e1, cycle_momentum=False)
schedulerG = CyclicLR(optimizer=optimizerG, base_lr=learning_rate,
                      max_lr=learning_rate*1e1, cycle_momentum=False)

# Adds spectral Norm to enforce the Lipstenstein Constraint to the network D.
for p in D.parameters():
    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
        p = nn.utils.spectral_norm(p)

# Prepares logging
log_dir = f"logs/{run_ID}"
os.makedirs(log_dir, exist_ok=True)
logger = SummaryWriter(log_dir)
tracked_images = next(iter(photo_dataloader))[:batch_size].to(device)
logger.add_image("Training/Tracked images",
                 vutils.make_grid(inv_normalize(tracked_images)), 0)

save_dir = f"./checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Endless range loop
for epoch in itertools.count():

    for i, (photo_batch_batch, cartoon_batch_batch) in tqdm(enumerate(zip(photo_dataloader, cartoon_dataloader)), desc=f"Training epoch {epoch}"):
        # [niter*bs, C, H, W] -> [niter, bs, C, H W]
        photo_batch_batch = torch.chunk(photo_batch_batch, n_critic_iters + 1)
        cartoon_batch_batch = torch.chunk(cartoon_batch_batch, n_critic_iters)

        ###############
        ### Train D ###
        ###############
        D.train()
        G.eval()
        for p in D.parameters():
            p.requires_grad = True
        for p in G.parameters():
            p.requires_grad = False

        for photo_batch, cartoon_batch in zip(photo_batch_batch, cartoon_batch_batch):
            photo_batch = photo_batch.to(device)
            cartoon_batch = cartoon_batch.to(device)

            D.zero_grad()
            with torch.cuda.amp.autocast():
                generated_cartoon_batch = G(photo_batch)

                real_pred = D(cartoon_batch)
                fake_pred = D(generated_cartoon_batch)

                loss_D = adv_loss(real_pred, fake_pred).sum()
            scaler_D.scale(loss_D).backward()
            # TODO: Possibly add GP
            scaler_D.step(optimizer=optimizerD)
            scaler_D.update()

            schedulerD.step()

        ###############
        ### Train G ###
        ###############

        # Remaining item is used to train G.
        photo_batch = photo_batch_batch[-1].to(device)

        D.eval()
        G.train()
        for p in D.parameters():
            p.requires_grad = False
        for p in G.parameters():
            p.requires_grad = True

        G.zero_grad()
        with torch.cuda.amp.autocast():
            generated_cartoon_batch = G(photo_batch)
            fake_pred = D(generated_cartoon_batch)
            loss_G = BCE_loss(fake_pred, cartoon_labels) + \
                content_loss(generated_cartoon_batch, photo_batch)
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optimizer=optimizerG)
        scaler_G.update()

        schedulerG.step()

        ###############
        ### Logging ###
        ###############
        if i % log_interval == 0:
            global_step = epoch * \
                min(len(photo_dataloader), len(cartoon_dataloader)) + i
            logger.add_scalar("Training/Loss D", loss_D, global_step)
            logger.add_scalar("Training/Loss G", loss_G, global_step)
            with torch.no_grad():
                logger.add_image("Training/Generated images",
                                 vutils.make_grid(inv_normalize(G(tracked_images))), global_step)

    torch.save({
        "D": D.state_dict(),
        "G": G.state_dict(),
        "OptimizerD": optimizerD.state_dict(),
        "OptimizerG": optimizerG.state_dict(),
        "SchedulerD": schedulerD.state_dict(),
        "SchedulerG": schedulerG.state_dict(),
        "Epoch": epoch
    }, os.path.join(save_dir, f"{run_ID}.pth"))
