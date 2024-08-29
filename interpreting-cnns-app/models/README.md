# Models Directory

This directory is intended to store scripts, model architectures, trained models, and evaluation results for the project aimed at segmenting vessels from medical images using a CNN model. This project focuses on analyzing the CNN's predictions by evaluating its gradient-based explanations.

## Folder Structure

- `evaluation-results/`:

  - Stores evaluation metrics and visual results from training sessions.
  - `vess_map_custom_cnn.png`: An example plot showing the training and validation loss or other performance metrics of the custom CNN model.

- `trained-models/`:

  - Contains the trained model weights.
  - `vess_map_custom_cnn.pth`: The saved state of the trained custom CNN model. Note: The models trained for this project are larger than GitHub's 100MB size limit, so the actual model files may not be included here but should be available separately for download.

- `train.py`:

  - A script to train the custom CNN model on the dataset of vessel images and their corresponding masks.

- `vessel_training_evaluation.ipynb`:

  - A Jupyter notebook for training and evaluating the model. It provides a more interactive approach to model training, visualization, and analysis.

- `vessel_training_utils.py`:

  - Contains utility functions required for training and evaluating the model, including functions to calculate metrics, handle regularization, and visualize results.

- `vess_map_custom_cnn.py`:
  - Defines the architecture of the custom CNN model, which is a simplified version of a ResNet designed for vessel segmentation tasks.

## Tips for Adapting Scripts and Models for Other Projects

### train.py

This script sets up the data, defines the model architecture, and runs the training loop for the CNN model.

- **Changing Dataset**: Update the paths for `image_dir`, `mask_dir`, and `skeleton_dir` to point to the directories containing your dataset. Ensure these directories have images and corresponding masks suitable for segmentation tasks.
- **Modifying Hyperparameters**: Adjust hyperparameters such as `batch_size`, `num_epochs`, `learning_rate`, `alpha_l1`, `alpha_l2`, and `weight_decay` to suit your dataset size and model complexity.
- **Regularization and Scheduler**: Experiment with different regularization modes ('l1', 'l2', or 'none') and learning rate schedules to optimize model training for your specific problem.
- **Model Architecture**: If you need a more complex model, consider adding more layers or using a pre-trained model with fine-tuning. Adjust the `CustomResNet` initialization accordingly.

### vess_map_custom_cnn.py

This file defines the `CustomResNet` model, which is based on a simplified version of ResNet architecture.

- **Model Complexity**: To adapt this script to different hardware capabilities or dataset requirements, modify the number of layers, channels, or the inclusion of dropout layers. If running on hardware with more memory, increase the depth or width of the model.
- **Input Channels**: Modify the first convolution layer if working with colored images (change `in_channels=1` to `in_channels=3`).
- **Number of Classes**: Update the `num_classes` parameter in the final layer if your dataset contains more than two classes (e.g., multiple types of vessels).

### vessel_training_utils.py

Contains utility functions for training, including visualization, calculating IoU, and plotting metrics.

- **Adding New Metrics**: Extend the `iou_metric` function or add new functions to compute other segmentation metrics such as Dice coefficient or precision/recall.
- **Visualization**: Update the `show_images` function to visualize different kinds of data (e.g., multi-channel images). Adjust the number of subplots and their content accordingly.
- **Data Augmentation**: Modify the augmentation strategies used in `apply_transform` within the dataset loader to include more variations like rotation, brightness adjustment, or contrast changes.

### vessel_training_evaluation.ipynb

A Jupyter notebook that integrates the training, validation, and visualization functions.

- **Interactive Analysis**: Use this notebook to experiment with different hyperparameters or training settings. It allows for easy visualization of intermediate results, which is useful for debugging and model analysis.

## Notes

- **Large Model Files**: Trained models exceeding 100MB are not included in the repository due to GitHub's size limits.
