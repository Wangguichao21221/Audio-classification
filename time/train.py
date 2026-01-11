import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from model import OneDCNN
from generate import MFCCDataset
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import utils
# 分割 train_dataset 为训练集和验证集
def split_train_val(dataset, val_ratio=0.1, random_state=42):
    # 提取所有样本的索引和对应标签（后者用于 stratify 分层抽样）
    indices = np.arange(len(dataset))
    labels = [label for _, label in dataset]  # 获取所有样本的标签

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,  # val_ratio 决定验证集大小
        stratify=labels,      # 保持类别分布相同
        random_state=random_state
    )

    # 使用 Subset 创建训练集和验证集
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset
def collate_fn(batch):
    X, y = zip(*batch)
    X = torch.stack(X)
    y = torch.tensor([label_to_index[label] for label in y])
    return X, y


TRAIN_NPY_DIR = "./train_npy"
TEST_NPY_DIR = "./test_npy"
SUBMIT_CSV_PATH = "./submit_time.csv"
val_ratio = 0.2 
# Prepare training dataset
train_dataset = MFCCDataset(data_dir=TRAIN_NPY_DIR, is_test=False)
test_dataset = MFCCDataset(data_dir=TEST_NPY_DIR, is_test=True)

train_subset, val_subset = split_train_val(train_dataset, val_ratio=val_ratio)
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, collate_fn=collate_fn)

label_to_index = {label: i for i, label in enumerate(sorted(set(label for _, label in train_dataset)))}
index_to_label = {i: label for label, i in label_to_index.items()}


# Training without K-Fold
def train_model(train_loader,val_loader,lr = 1e-4,num_epochs = 150):
    test_predictions = np.zeros((len(test_dataset), len(label_to_index)))
    print("\n==== Training ====\n")

    model = OneDCNN(input_shape=(40, 344), num_classes=len(label_to_index))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    train_acc_per_epoch = []
    val_acc_per_epoch = []
    for epoch in range(1, num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # print(X_batch.shape,y_batch.shape)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y_batch.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                _, predicted = outputs.max(1)
                val_correct += (predicted == y_val).sum().item()
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * y_val.size(0)
                val_total += y_val.size(0)
        val_loss=val_loss/val_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch} - Train Loss: {train_loss/total:.4f}, Train Acc: {train_acc:.4f},Val loss: {val_loss:.4f} ,Val Acc: {val_acc:.4f}")
        train_loss_per_epoch.append(train_loss/total)
        val_loss_per_epoch.append(val_loss)
        train_acc_per_epoch.append(train_acc)
        val_acc_per_epoch.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    print(f"Training completed. Best Validation Accuracy: {best_val_acc:.4f}")
    utils.plot_training_metrics_complete(train_loss_per_epoch,val_loss_per_epoch,train_acc_per_epoch,val_acc_per_epoch,'./logs')
    
if __name__ == "__main__":
    # Choose one of the two functions below based on your training approach
    # train_model()  # Uncomment this line for K-Fold training
    os.makedirs('./logs',exist_ok=True)
    train_model(train_loader,val_loader,num_epochs=200,lr = 1e-3)  # Uncomment this line for no K-Fold training