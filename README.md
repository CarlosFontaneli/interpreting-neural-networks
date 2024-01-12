# Blood Vessel Project: Analysis and Interpretation through Neural Networks

Blood vessels are crucial for vertebrate organisms, transporting nutrients and oxygen to tissues and cells. This project focuses on the computational analysis of these structures in images, particularly in interpreting the results obtained through Convolutional Neural Networks (CNNs).

## Context

### Importance of Blood Vessels

Blood vessels have profound implications in both medical diagnostics and neuroscience research. Advanced studies involve topics like the Blood-Brain Barrier, which directly influences the relationship between blood vessels and neurons.

### Image Segmentation and Detection

Recent advances in image processing and machine learning have facilitated the segmentation of blood vessels in digital images, with CNNs showing promising results.

## Challenges

Neural networks, despite being powerful, are often considered as "black boxes," making it difficult to interpret the results. This contrasts with more traditional approaches that are more intuitive. This project seeks to analyze the interpretability of the results and find ways to improve reliability in automated detection.

## Objectives

- Analyze which pixels in an image impact the segmentation of regions of interest.
- Investigate the image information that influences the segmentation result.
- Apply neural network interpretation techniques to understand and optimize the process.

## Future Directions

This research is expected to aid in developing more reliable techniques for analyzing blood vessel images. Additionally, the project may contribute to training neural networks with less data.

## Repository Structure and Contents

- **Attributions Maps**: Contains influence maps of an analysis of pixel/region of influence (positive and negative) of the final output of CNNs made with methods of Captum and with gradient analysis.

  - `classification/`: Deep analysis with Captum for the influence of pixels in the final classification of a CNN model, including analysis of the influence of pixels for the predicted class and the opposite class.
  - `digits/`: Contains an analysis of pixel influence on the MNIST dataset.
  - `segmentation/`: Analysis for CNNs set to segmentation tasks, with Captum and gradient analysis. Includes models with and without batch normalization, for images with and without augmentation (a black dot in the middle of the image).
  - `vessels/`: Similar to segmentation but for a more specific dataset.

- **Data**: This directory includes datasets like VessMap (images, labels, skeletons), classification, and segmentation custom datasets. Also includes classes for loading and setting Pytorch`s DataLoader of the datasets for training and validation of the models and for the experiments.

- **Experiments**: Contains analysis and experiments with Captum and its methods, gradient analysis, model precision with Intersection over Union (IoU), metrics, and sparsity analysis for the models, along with additional experiments.

- **Models**: This folder contains definitions of classes of custom neural networks based on PyTorch, and many models trained in different states, datasets, and training methods.

- **Train Models Notebooks**: Jupyter notebooks that contain code for training, validating, and testing the neural network models used in the project.
