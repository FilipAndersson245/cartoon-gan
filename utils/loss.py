import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import gc

class AdversialLoss(nn.Module):
    def __init__(self, cartoon_labels, fake_labels):
        super(AdversialLoss, self).__init__()
        self.cartoon_labels = cartoon_labels
        self.fake_labels = fake_labels
        self.base_loss = nn.BCEWithLogitsLoss()

    def forward(self, cartoon, generated_f, edge_f):
        #print(cartoon.shape, self.cartoon_labels.shape)
        D_cartoon_loss = self.base_loss(cartoon, self.cartoon_labels)
        D_generated_fake_loss = self.base_loss(generated_f, self.fake_labels)
        D_edge_fake_loss = self.base_loss(edge_f, self.fake_labels)

        # TODO Log maybe?
        return D_cartoon_loss + D_generated_fake_loss + D_edge_fake_loss
        

class ContentLoss(nn.Module):
    def __init__(self, omega=10):
        super(ContentLoss, self).__init__()

        self.base_loss = nn.L1Loss()
        self.omega = omega

        perception = list(vgg16(pretrained=True).features)[:25]
        self.perception = nn.Sequential(*perception).eval()

        for param in self.perception.parameters():
            param.requires_grad = False

        gc.collect()

    def forward(self, x1, x2):
        x1 = self.perception(x1)
        x2 = self.perception(x2)
        
        return self.omega * self.base_loss(x1, x2)
        
