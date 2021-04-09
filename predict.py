from PIL import Image
import argparse
import os
import mimetypes
from utils.transforms import get_no_aug_transform
import torch
from models.generator import Generator
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import cv2
from torchvision import utils as vutils
import subprocess
import tempfile
import re
from tqdm import tqdm
import time


def inv_normalize(img):
    # Adding 0.1 to all normalization values since the model is trained (erroneously) without correct de-normalization
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    img = img.clamp(0, 1)
    return img

def predict_images(image_list):
    trf = get_no_aug_transform()
    image_list = torch.from_numpy(np.array([trf(img).numpy() for img in image_list])).to(device)

    with torch.no_grad():
        generated_images = netG(image_list)
    generated_images = inv_normalize(generated_images)

    pil_images = []
    for i in range(generated_images.size()[0]):
        generated_image = generated_images[i].cpu()
        pil_images.append(TF.to_pil_image(generated_image))
    return pil_images

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]

def predict_file(input_path, output_path):
    # File is image
    if mimetypes.guess_type(input_path)[0].startswith("image"):
        image = Image.open(input_path).convert('RGB')
        predicted_image = predict_images([image])[0]
        predicted_image.save(output_path)
    # File is video
    elif mimetypes.guess_type(input_path)[0].startswith("video"):
        # Create temp folder for storing frames as images
        temp_dir = tempfile.TemporaryDirectory()
        # Extract frames from video
        subprocess.run(f"ffmpeg -i \"{input_path}\" -loglevel error -stats \"{os.path.join(temp_dir.name, 'frame_%07d.png')}\"")
        # Process images with model
        frame_paths = listdir_fullpath(temp_dir.name)
        batches = [*divide_chunks(frame_paths, batch_size)]
        for path_chunk in tqdm(batches):
            imgs = [Image.open(p) for p in path_chunk]
            imgs = predict_images(imgs)
            for path, img in zip(path_chunk, imgs):
                img.save(path)
        # Get video frame rate
        frame_rate = subprocess.check_output(f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate \"{input_path}\"")
        frame_rate = eval(frame_rate.split()[0]) # Dirty eval
        # Combine frames with original audio
        subprocess.run(f"ffmpeg -y -r {frame_rate} -i \"{os.path.join(temp_dir.name, 'frame_%07d.png')}\" -i \"{input_path}\" -map 0:v -map 1:a? -loglevel error -stats \"{output_path}\"")
    else:
        raise IOError("Invalid file extension.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="This file is used to convert images/videos to cartoons.")

    parser.add_argument("-i", "--input", type=str,  required=True, help="Path to file (image/video) or path to folder containing multiple images.")
    parser.add_argument("-o", "--output", type=str,  required=True, help="Where predicted images/videos should be saved. If --input is a single file, --output should be a single file as well.")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-b", "--batch_size", type=int, default=4)

    input_path, output_path, user_stated_device, batch_size = vars(parser.parse_args()).values()
    
    device = torch.device(user_stated_device)
    pretrained_dir = "./checkpoints/trained_netG.pth"
    netG = Generator().to(device)
    netG.eval()

    # Load weights
    if user_stated_device == "cuda":
        netG.load_state_dict(torch.load(pretrained_dir))
    else:
        netG.load_state_dict(torch.load(pretrained_dir, map_location=torch.device('cpu')))

    # Single file
    if os.path.isfile(input_path):
        predict_file(input_path, output_path)
    # Multiple files
    else:
        os.makedirs(output_path, exist_ok=True)
        for file_name in tqdm(os.listdir(input_path), desc="Processing files"):
            file_path = os.path.join(input_path, file_name)
            output_file_path = os.path.join(output_path, file_name)
            predict_file(file_path, output_file_path)
