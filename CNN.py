import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PneumoniaCNN(nn.Module):
    """
    Improved CNN for melanoma classification with advanced features.
    - More convolutional layers and increased filter sizes.
    - Use of pretrained layers (transfer learning).
    - Learning rate scheduler support.
    """

    def __init__(self, pretrained=False):
        """Initialize the network layers."""
        super().__init__()

        # Pretrained layers (using a part of a ResNet or VGG model)
        if pretrained:
            # For example, you can load ResNet18 pretrained on ImageNet
            self.mobilenet = models.mobilenet(pretrained=True)
            self.mobilenet.fc = nn.Linear(self.resnet.fc.in_features, 2)  
        else:
            # Convolutional layers with Batch Normalization
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)

            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)

            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)

            # Global Average Pooling
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            # Fully connected layers with Dropout
            self.fc1 = nn.Linear(128, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 224, 224)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        if hasattr(self, 'mobilenet'):  # If using pre-trained model
            return self.mobilenet(x)

        # Convolutional block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Convolutional block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Convolutional block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)

        return x  # Raw logits (no softmax, use CrossEntropyLoss during training)

    def get_feature_dims(self, input_size=(1, 224, 224)):
        """
        Calculate feature dimensions at each layer for debugging.

        Args:
            input_size (tuple): Input dimensions (channels, height, width)

        Returns:
            dict: Dictionary containing feature dimensions at each layer
        """
        dims = {}
        x = torch.zeros(1, *input_size)  # Create dummy input

        # Conv1
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
        dims['conv1'] = x.shape

        # Conv2
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
        dims['conv2'] = x.shape

        # Conv3
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))
        dims['conv3'] = x.shape

        # Global Average Pooling
        x = self.global_pool(x)
        dims['global_pool'] = x.shape

        # Flattened
        dims['flatten'] = x.view(x.size(0), -1).shape

        return dims
