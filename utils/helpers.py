import torch

def unnormalize(t):
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(t.device)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(t.device)
    return t * std + mean