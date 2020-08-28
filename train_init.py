import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
import torchvision.utils as vutils
from torchvision.models import vgg16

from utils.transforms import get_photo_train_loader
from torch.utils.tensorboard import SummaryWriter
from models.generator import Generator

from datetime import datetime
import os
import itertools
import gc
from tqdm import tqdm

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
run_ID = datetime.now().strftime("%Y.%m.%d %H.%M") + "_init"
batch_size = 20
image_size = 224
learning_rate = 1.5e-4
beta1, beta2 = (.5, .99)
weight_decay = 1e-4
epochs = 100
log_interval = 5


def inv_normalize(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    img = img.clamp(0, 1)
    return img


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
BCE_loss = nn.BCEWithLogitsLoss().to(device)

photo_dataloader = get_photo_train_loader(
    224, batch_size=batch_size)


scaler_G = torch.cuda.amp.GradScaler()

# Models
G = Generator().to(device)

optimizerG = AdamW(G.parameters(), lr=learning_rate,
                   betas=(beta1, beta2), weight_decay=weight_decay)

schedulerG = CyclicLR(optimizer=optimizerG, base_lr=learning_rate,
                      max_lr=learning_rate*1e1, cycle_momentum=False)

# Prepares logging
log_dir = f"logs/{run_ID}"
os.makedirs(log_dir, exist_ok=True)
logger = SummaryWriter(log_dir)
tracked_images = next(iter(photo_dataloader)).to(device)
logger.add_image("Init training/Tracked images",
                 vutils.make_grid(inv_normalize(tracked_images)), 0)

save_dir = f"./checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Endless range loop
for epoch in itertools.count():

    for i, photo_batch in tqdm(enumerate(photo_dataloader), total=len(photo_dataloader), desc=f"Training epoch {epoch}"):

        ###############
        ### Train G ###
        ###############
        photo_batch = photo_batch.to(device)

        G.train()
        for p in G.parameters():
            p.requires_grad = True

        G.zero_grad()
        with torch.cuda.amp.autocast():
            generated_cartoon_batch = G(photo_batch)
            loss_G = content_loss(generated_cartoon_batch, photo_batch)
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optimizer=optimizerG)
        scaler_G.update()

        schedulerG.step()

        ###############
        ### Logging ###
        ###############
        if i % log_interval == 0:
            global_step = epoch * len(photo_dataloader) + i
            logger.add_scalar("Init training/Loss G", loss_G, global_step)
            with torch.no_grad():
                logger.add_image("Init training/Generated images",
                                 vutils.make_grid(inv_normalize(G(tracked_images))), global_step)

    torch.save({
        "G": G.state_dict(),
        "OptimizerG": optimizerG.state_dict(),
        "SchedulerG": schedulerG.state_dict(),
        "Epoch": epoch
    }, os.path.join(save_dir, f"{run_ID}.pth"))
