# Fruit image source: https://github.com/Horea94/Fruit-Images-Dataset
# Code based on Heartbeat's PyTorch Guide
# URL: https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864

import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self, num_classes):
        super(BasicModel, self).__init__()

        # define the layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 1024)  # reshaping
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

