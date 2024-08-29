import sys
import torch
import torch.nn as nn
import torch.optim as optim
from vessel_training_utils import iou_metric, sum_params, training_loop, save_model
from vess_map_custom_cnn import CustomResNet

sys.path.append("../data/")
from vess_map_dataset_loader import vess_map_dataloader

# Setting up GPU
torch.cuda.empty_cache()
device = torch.device("cuda")

# Data directories
image_dir = "../data/VessMap/images"
mask_dir = "../data/VessMap/labels"
skeleton_dir = "../data/VessMap/skeletons"

# DataLoader setup
batch_size = 10
train_size = 0.8
image_size = 64

train_loader, test_loader = vess_map_dataloader(
    image_dir, mask_dir, skeleton_dir, batch_size, train_size, image_size
)

# Initialize metrics
train_losses = []
train_aux_losses = []
test_losses = []
test_accuracies = []
test_ious = []

# Hyperparameters
alpha_l1 = 0.00001
alpha_l2 = 0.00001
regularization_mode = "none"  # Options: 'none', 'l1', 'l2'
num_epochs = 200
weight_decay = 1e-4

# Model, criterion, optimizer, and scheduler
model = CustomResNet(num_classes=2).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=10, factor=0.5, verbose=True
)

# Training the model
training_loop(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    regularization_mode=regularization_mode,
    alpha_l1=alpha_l1,
    alpha_l2=alpha_l2,
    train_losses=train_losses,
    train_aux_losses=train_aux_losses,
    test_losses=test_losses,
    test_accuracies=test_accuracies,
    test_ious=test_ious,
)

# Save the trained model
save_model(
    model,
    f"../models/vess_map_regularized_{regularization_mode}_{num_epochs}.pth",
    regularization_mode,
)

print(
    f"Model training completed and saved as vess_map_regularized_{regularization_mode}_{num_epochs}.pth"
)
