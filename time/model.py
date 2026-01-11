import torch
import torch.nn as nn
import torch.nn.functional as F

class OneDCNN(nn.Module):
    def __init__(self, input_shape=(40, 344), num_classes=20, dropout_rate=0.1):
        super(OneDCNN, self).__init__()
        
        # 核心：整网用单个Sequential封装，包含所有层（卷积+BN+池化+Dropout+全连接）
        self.model = nn.Sequential(
            # 卷积块1 + Dropout1d
            nn.Conv1d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout1d(dropout_rate),  # 卷积层后加1D Dropout
            
            # 卷积块2 + Dropout1d
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout1d(dropout_rate),  # 卷积层后加1D Dropout
            
            # 卷积块3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 全局平均池化 + 展平
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),  # 替代torch.flatten，适配Sequential
            
            # 全连接层 + Dropout（2D Dropout，适配全连接）
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channel, sequence)
        return self.model(x)
if __name__ == "__main__":
    model = OneDCNN()