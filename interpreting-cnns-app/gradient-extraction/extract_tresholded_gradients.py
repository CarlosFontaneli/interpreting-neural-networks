# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


# Function to process each pixel for thresholding
def process_pixel(args):
    gradient, threshold_proportion, i, j = args
    threshold = np.max(gradient) * threshold_proportion
    mask = np.abs(gradient[i, j]) > threshold
    num_pixels_above_threshold = np.sum(mask)
    if j == 5:
        exit
    return (i, j, num_pixels_above_threshold)


# Function to create thresholded images from gradients
def create_thresholded_images(gradient, threshold_proportion=0.01):
    gradient_np = gradient.detach().cpu().numpy()
    thresholded_image = np.zeros_like(gradient_np[0, 0])

    # Create a list of arguments for each pixel
    args_list = [
        (gradient_np, threshold_proportion, i, j)
        for i in range(gradient_np.shape[0])
        for j in range(gradient_np.shape[1])
    ]

    # Use multiprocessing to process each pixel in parallel
    with Pool() as pool:
        results = pool.map(process_pixel, args_list)

    # Fill the thresholded_image with the results
    for i, j, num_pixels_above_threshold in results:
        thresholded_image[i, j] = num_pixels_above_threshold

    # Normalize the thresholded_image
    thresholded_image = (thresholded_image - thresholded_image.min()) / (
        thresholded_image.max() - thresholded_image.min()
    )
    return thresholded_image


# Main function to handle processing and saving of thresholded images
def main():
    for idx in range(5):
        loaded_gradient = torch.load(f"./gradients/jacobian_gradient_{idx}.pt")
        print(f"Gradient {idx} loaded, shape of: {loaded_gradient.shape}")

        # Create thresholded image
        img = create_thresholded_images(loaded_gradient)

        # Save the thresholded image as a .npy file
        np.save(f"./thresholded_gradients/image_{idx}.npy", img)
        print(f"Thresholded gradient {idx} saved!")


if __name__ == "__main__":
    main()
