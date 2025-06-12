import torch.nn as nn
import torch.nn.functional as F
import torch

class WakeWordModel(nn.Module):
    def __init__(self, input_shape):
        super(WakeWordModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape).unsqueeze(1)  # (1, 1, time, n_mfcc)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)