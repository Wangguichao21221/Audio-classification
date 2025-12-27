import numpy as np
import torch
import os
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
# 1. 加载测试数据
device ='cpu'
test_X = np.load('./fusai_test_mfcc_128.npy', allow_pickle=True)
print(test_X.shape)
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5,
              'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream': 11,
              'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17,
              'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}



# 2. 重塑为PyTorch格式: (batch, channels, length)
test_X = test_X[:, np.newaxis, :]  # 添加通道维度

# 3. 加载模型
from train import Conv1DModel,AudioDataset
model = Conv1DModel(input_dim=128, num_classes=20)

# 4. 加载权重
checkpoint = torch.load('./99.8571_weights_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

X_test = test_X
test_dataset = AudioDataset(X_test, np.zeros(len(X_test))) 
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

test_correct = 0
test_total = 0
predictions = []
with torch.no_grad():
    for data, target in test_loader:
        data= data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(label_dict_inv[predicted.item()])


files = np.load("./fusai_test_filenames.npy",allow_pickle=True)

# 5. 预测所有样本


with open('submission.csv', 'w') as f:
    f.write("name,label\n")
    for filename, label in zip(files, predictions):
        f.write(f'{filename},{label}\n')
print(f"总共预测 {len(predictions)} 个样本")