# Data Directory

This directory contains datasets and and data loaders used for my scientific initiation project that aims to analyze CNN model predictions for the segmentation of vessels from medical images. The goal is to explain the model's predictions based on the analysis of its gradients.

## Folder Structure

- `cropped_images/`:

  - Contains cropped sample images in `.png` format. These images are used for model training and validation.

- `VessMap/`:

  - This folder contains subfolders for raw images, labels, and skeletons used in the project.
  - `images/`: Contains raw `.tiff` images representing the vessel maps from medical images.
  - `labels/`: Contains corresponding ground truth `.png` label images for the vessel maps, which are used as masks in model training.
  - `skeletons/`: Contains skeletonized versions of the vessel maps in `.png` format, which are used for further analysis.
  - `VessMAP.zip`: Compressed file containing the original dataset.

- `vess_map_dataset_loader.py`:

  - Script that provides functionality for loading and preprocessing the dataset using PyTorch's `DataLoader`.
  - It defines a `VessMapDataset` class for handling the images, labels, and skeletons, and applying random transformations for data augmentation.
  - The script includes a `vess_map_dataloader` function that splits the dataset into training and testing sets and returns corresponding data loaders.

## Dataset Adaptation Tips

- **Changing Image Directory**: To adapt this project to different datasets, modify the `image_dir`, `mask_dir`, and `skeleton_dir` parameters in the `VessMapDataset` class to point to new directories containing your images. Ensure that each image has a corresponding mask for the segmentation task.
- **Different Image Formats**: If your dataset uses formats other than `.png` or `.tiff`, adjust the file extensions accordingly in the `_find_pairs` method of the `VessMapDataset` class. This ensures the correct matching of images and masks.
- **Alter Data Augmentation**: Customize the transformations (e.g., rotations, scaling, color jitter) in the `apply_transform` method to improve model robustness and simulate variations seen in images in the context of your project. These augmentations can help the model generalize better to different patient data.
- **Image Size**: Update the `image_size` parameter to match the dimensions of the new dataset's images. This can be done when initializing the `VessMapDataset` instance or calling the `vess_map_dataloader` function. Consistent image sizes are crucial for CNNs to process the images correctly.
- **Ensure Pairing of Images and Masks**: For accurate model training, each image in your dataset should have a corresponding mask file that marks the segmented vessels. The file names should match appropriately to ensure that the loader pairs them correctly.
