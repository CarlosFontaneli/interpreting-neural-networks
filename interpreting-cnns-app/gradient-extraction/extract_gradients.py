import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
import time

sys.path.append("../models/")
from vess_map_custom_cnn import CustomResNet


# Function to wrap model to return probabilities for only the vessel channel
def wrapper(model):
    """Wrap model to return probabilities and only the vessel channel."""

    def new_model(img):
        out = model(img)
        probs = F.softmax(out, dim=1)
        return probs[:, 1]

    return new_model


# Function to get all gradients
def get_all_gradients(model, image, sampling_rate=1, device="cuda", vectorize=False):
    model.to(device)
    model.eval()
    model_wrapped = wrapper(model)

    sampled_image = image[:, ::sampling_rate, ::sampling_rate]
    sampled_image = sampled_image.to(device).unsqueeze(0)
    sampled_image.requires_grad = True

    jacobian = torch.autograd.functional.jacobian(
        model_wrapped, sampled_image, vectorize=vectorize
    )
    jacobian = jacobian.squeeze().to("cpu")
    return jacobian


# Function to load images from a directory
def load_images_from_directory(directory_name):
    image_files = sorted(os.listdir(directory_name))
    images = []
    for file_name in image_files:
        if file_name.endswith(".png"):
            img_path = os.path.join(directory_name, file_name)
            img = Image.open(img_path)
            img_array = np.array(img) / 255.0
            images.append(img_array)
    return torch.tensor(np.array(images), dtype=torch.float).unsqueeze(1).to("cpu")


# Class for handling image conversion and state management
class ImageUF:
    """Represents a float tensor as uint8, storing the minimum and maximum values."""

    def __init__(self, image=None):
        if image is None:
            self.min = None
            self.max = None
            self.image_uint8 = None
        else:
            min_val = image.min()
            max_val = image.max()
            image_norm = 255.0 * (image - min_val) / (max_val - min_val)
            self.image_uint8 = image_norm.round().to(torch.uint8)
            self.min = min_val
            self.max = max_val

    def state_dict(self):
        return {"min": self.min, "max": self.max, "image_uint8": self.image_uint8}

    def load_state_dict(self, state):
        self.min = state["min"]
        self.max = state["max"]
        self.image_uint8 = state["image_uint8"]

    def to_float(self):
        image_float32_norm = self.image_uint8.to(torch.float32)
        return image_float32_norm * (self.max - self.min) / 255.0 + self.min


# Main function to handle gradient extraction
def main():
    # Load images
    original_images = load_images_from_directory("../data/cropped_images")

    # Load the pre-trained model
    model_weighted = CustomResNet(num_classes=2)
    model_weighted.load_state_dict(
        torch.load(f"../models/trained-models/vess_map_custom_cnn.pth")
    )

    # Extract and save gradients for each image
    for idx in range(5):
        start_time = time.time()  # Record the start time before extracting the gradient

        gradient = get_all_gradients(
            model_weighted, original_images[idx], sampling_rate=1
        )
        torch.save(gradient, f"./gradients/jacobian_gradient_{idx}.pt")
        end_time = time.time()  # Record the end time after saving the gradient
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Gradient {idx} extraction and saving took {elapsed_time:.2f} seconds.")
    # Example of loading and handling gradients
    loaded_gradient = torch.load("./gradients/jacobian_gradient_0.pt")

    # Using ImageUF class for handling and saving tensor
    jacobian_uf = ImageUF(loaded_gradient)
    torch.save(jacobian_uf.state_dict(), "jacobian.pt")

    # Load data from disk
    jacobian_uf2 = ImageUF()
    jacobian_uf2.load_state_dict(torch.load("jacobian.pt"))

    # Convert back to float and compare
    jacobian_rec = jacobian_uf2.to_float()
    rel_diff = ((jacobian_rec - loaded_gradient) / loaded_gradient).abs()
    max_diff = rel_diff[loaded_gradient.abs() > 1e-2].max()
    print(f"Maximum relative difference: {max_diff}")


if __name__ == "__main__":
    main()
