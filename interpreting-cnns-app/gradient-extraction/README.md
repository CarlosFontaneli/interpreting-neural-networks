# Gradient Extraction Directory

This directory is dedicated to analyzing the gradients of trained CNN models for vessel segmentation tasks in medical images. The scripts and files here are used to extract, process, and validate the gradients of the models to gain insights into model predictions.

## Folder Structure

- `extract_gradients.py`:

  - A script that extracts all the gradients for a given input by calculating the Jacobian matrix.
  - It saves these gradients in the `gradients/` directory for further analysis.

- `extract_thresholded_gradients.py`:

  - This script processes the extracted gradients to create thresholded gradients. It reduces the information to a unique image by applying a threshold, highlighting significant gradient values.

- `gradients/`:

  - Stores the raw gradients extracted from the model for each input.
  - Files like `jacobian_gradient_0.pt`, `jacobian_gradient_1.pt`, etc., are PyTorch tensors containing the Jacobian matrices.

- `thresholded_gradients/`:

  - Contains thresholded gradient images, which are the processed versions of the raw gradients to highlight important features.
  - Files like `image_0.npy`, `image_1.npy`, etc., store these thresholded images.

- `validate_gradients.ipynb`:
  - A Jupyter notebook for validating the extracted gradients. It plots the original images, model's softmax output, and the extracted gradients side-by-side for visual analysis.

## Tips for Adapting Scripts for Other Projects

### extract_gradients.py

This script extracts gradients using the Jacobian matrix for given inputs to understand how changes in the input influence the model's output.

- **Model Adaptation**: Replace `CustomResNet` with your own model architecture. Ensure the model is loaded correctly and has compatible input and output shapes.
- **Image Sources**: Update the path in `load_images_from_directory` to point to the directory containing your dataset's images. Ensure the images are pre-processed appropriately (e.g., normalized) before being passed to the model.
- **Gradient Calculation**: Modify the `get_all_gradients` function if different gradient computation strategies are needed. The wrapper can be adjusted to focus on different output channels or layers.
- **Performance Optimization**: For larger models or datasets, consider using a reduced `sampling_rate` or splitting the dataset into batches for gradient extraction.

### extract_thresholded_gradients.py

This script takes the extracted gradients and creates a thresholded version to summarize the gradients into a single image, highlighting only the most significant gradients.

- **Thresholding Method**: Adjust the `threshold_proportion` parameter in the `create_thresholded_images` function to fine-tune how aggressively gradients are filtered. This can help focus on different levels of detail.
- **Processing Speed**: Utilize multiprocessing (as implemented) to speed up the processing of gradients, especially for high-resolution images or large datasets.
- **Visualization**: Modify the script to save visualizations directly (using `matplotlib`) for each thresholded gradient, which can be helpful for batch analysis and comparison.

### validate_gradients.ipynb

This notebook is used for visually validating the extracted gradients by comparing them with the original input images and the model's softmax output.

- **Model and Data Loading**: Ensure the model and data loading sections are updated to reflect your own model architecture and data paths.
- **Custom Visualization**: Adapt the plots to include more specific comparisons, such as overlaying gradients on the original images or comparing different gradient extraction methods.
- **Interactive Analysis**: Use this notebook to experiment with different threshold values or gradient extraction techniques interactively, providing immediate visual feedback.

## General Tips for Gradient Analysis

1. **Choosing the Right Inputs**: Select images that are representative of the various cases the model handles (e.g., clear vessels, occluded vessels) to gain insights into model behavior.
2. **Scalability**: If working with a large dataset, automate the gradient extraction and thresholding process, and store results efficiently for later analysis.
3. **Use GPU Acceleration**: Ensure that all gradient computations are performed on the GPU to reduce processing time. This can be managed by setting the `device` parameter to `"cuda"` where applicable.
4. **Model Interpretation**: Use the extracted gradients to interpret the model's decision-making process. Areas with high gradient values indicate regions of the input that strongly influence the model's output, providing insights into what the model considers important.
