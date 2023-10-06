# Imports
import torch.nn.functional as F
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, channels, kernel_size):
        super(CustomCNN, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(1, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # Second block
        self.conv2 = nn.Conv2d(channels, 2 * channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(2 * channels)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # Third block
        self.conv3 = nn.Conv2d(2*channels, 4 * channels, kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm2d(4 * channels)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        # Fourth block (no pooling after this)
        self.conv4 = nn.Conv2d(4*channels, 8 * channels, kernel_size, padding=kernel_size//2)
        self.bn4 = nn.BatchNorm2d(8 * channels)
        self.relu4 = nn.ReLU()
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(8*channels, 128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # 2 classes for Custom Dataset

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.drop1(F.relu(self.fc1(x)))
        x = self.fc2(x) 
        
        return x
