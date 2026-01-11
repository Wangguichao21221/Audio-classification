import torch.nn as nn
class Conv1DModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=20):
        super(Conv1DModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 第一卷积块
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
class MLP_4(nn.Module):
    def __init__(self, input_dim=128, num_classes=20):
        super(MLP_4, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        x = self.mlp(x)
        return x
class MLP_3(nn.Module):
    def __init__(self, input_dim=128, num_classes=20):
        super(MLP_3, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.mlp(x)
        return x
class MLP_2(nn.Module):
    def __init__(self, input_dim=128, num_classes=20):
        super(MLP_2, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.mlp(x)
        return x
class MLP_1(nn.Module):
    def __init__(self, input_dim=128, num_classes=20):
        super(MLP_1, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.mlp(x)
        return x
