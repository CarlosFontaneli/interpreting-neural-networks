# Imports
import torch
import torch.nn.functional as F

import numpy as np
import os
from PIL import Image

import sys

sys.path.append("../data/")
from VessMapDatasetLoader import vess_map_dataloader

torch.cuda.empty_cache()
device = torch.device("cuda")

# Dataloaders
image_dir = '/home/fonta42/Desktop/interpretacao-redes-neurais/data/VessMap/images'
mask_dir = '/home/fonta42/Desktop/interpretacao-redes-neurais/data/VessMap/labels'
skeleton_dir = '/home/fonta42/Desktop/interpretacao-redes-neurais/data/VessMap/skeletons'

batch_size = 10
train_size = 0.8

train_loader, test_loader = vess_map_dataloader(image_dir, 
                                  mask_dir, 
                                  skeleton_dir, 
                                  batch_size,
                                  train_size = train_size)

all_images = []

# Iterate through the entire train_loader
for batch in train_loader:
    images, _, _ = batch
    images = images.to(device)
    all_images.extend(images)
    
for batch in test_loader:
    images, _, _ = batch
    images = images.to(device)
    all_images.extend(images)
    
from torchvision.transforms.functional import to_pil_image

from torchvision.transforms.functional import to_pil_image, crop
import os
import torchvision.transforms.functional as TF

def save_cropped_images(image_array, directory_name, crop_size=(64, 64)):
    # Create the directory if it does not exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # Loop through the images, crop, and save them
    for i, img_tensor in enumerate(image_array):
        print(img_tensor.shape)
        # Calculate the top-left corner of the crop
        top = (img_tensor.size(1) - crop_size[0]) // 2
        left = (img_tensor.size(2) - crop_size[1]) // 2

        # Crop the tensor
        cropped_tensor = crop(img_tensor, top, left, crop_size[0], crop_size[1])

        # Convert the cropped tensor to a PIL image
        img = to_pil_image(cropped_tensor.squeeze().cpu())

        # Save the cropped image with reduced file size
        img.save(os.path.join(directory_name, f'image_{i}.png'), 'PNG', optimize=True, quality=20)

# Save all_images with cropping
save_cropped_images(all_images, './cropped_images')
