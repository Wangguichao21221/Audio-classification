import torch
import torch.nn as nn
import torch.nn.functional as F

class OneDCNN(nn.Module):
    def __init__(self, input_shape=(344, 40), num_classes=20):
        super(OneDCNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_shape[1],
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2)

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channel, sequence)

        x = F.relu(self.conv1(x))
        x = self.pool1(self.bn1(x))

        x = F.relu(self.conv2(x))
        x = self.pool2(self.bn2(x))

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
if __name__ == "__main__":
    model = OneDCNN()