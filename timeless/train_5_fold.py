from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
from model import Conv1DModel
import utils
# 5折交叉验证训练函数
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
def train_model(model, train_loader, test_loader,fold ,num_epochs=200, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    best_accuracy = 0.0
    best_model_path = None
    train_losses_per_epoch = []
    val_losses_per_epoch = []
    train_accuracy_per_epoch = []
    val_accuracy_per_epoch = []
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            # print(outputs.shape, target.shape)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() *target.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # 验证阶段
        model.eval()
        
        val_accuracy,val_loss = test_acc(model,device,test_loader)
        
        train_accuracy = 100.0 * train_correct / train_total
        train_loss == train_loss/train_total
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_path = f"pytorch_models/weights_best_{val_accuracy:.4f}_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
            }, best_model_path)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Val Loss: {val_loss/len(test_loader):.4f}, '
                  f'Val Acc: {val_accuracy:.2f}%')
        train_accuracy = train_accuracy/100.0
        val_accuracy = val_accuracy/100.0
        train_losses_per_epoch.append(train_loss)
        val_losses_per_epoch.append(val_loss)
        train_accuracy_per_epoch.append(train_accuracy)
        val_accuracy_per_epoch.append(val_accuracy)
    os.makedirs('./logs',exist_ok= True)
    os.makedirs(f'./logs/fold{fold}',exist_ok= True)
    utils.plot_training_metrics_complete(train_losses_per_epoch,val_losses_per_epoch,
                                         train_accuracy_per_epoch,val_accuracy_per_epoch,f'./logs/fold{fold}')
    return best_accuracy, best_model_path
def test_acc(model,device,test_loader):
    criterion = nn.CrossEntropyLoss()
    test_correct = 0
    test_total = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            total_loss += loss.item()*target.size(0)
    total_loss = total_loss / test_total
    test_accuracy = 100.0 * test_correct / test_total
    return test_accuracy,total_loss
def train_with_kfold(model_class, features, labels, num_classes, num_epochs=200, device='cuda', k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    best_models = []
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"\n==== Fold {fold + 1}/{k} ====")
        
        # 划分训练集和验证集
        X_train, X_val = features[train_idx], features[val_idx]
        Y_train, Y_val = labels[train_idx], labels[val_idx]
        
        # 数据形状 (channels, length)
        X_train = X_train.reshape(-1, 1, features.shape[1])
        X_val = X_val.reshape(-1, 1, features.shape[1])
        
        train_dataset = AudioDataset(X_train, Y_train)
        val_dataset = AudioDataset(X_val, Y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        # 初始化模型
        model = model_class(input_dim=features.shape[1], num_classes=num_classes)
        model.to(device)
        
        print(f"Fold [{fold + 1}] 模型参数总数: {sum(p.numel() for p in model.parameters())}")
        
        # 训练模型
        best_accuracy, best_model_path = train_model(model, train_loader, val_loader, fold= fold+1,num_epochs=num_epochs, device=device)
        print(f"Fold [{fold + 1}] Validation Accuracy: {best_accuracy:.2f}%")
        
        # 保存模型路径
        best_models.append(best_model_path)
        fold_accuracies.append(best_accuracy/100.0)
    
    print("\n==== 5折训练完成 ====")
    print(f"每折验证集准确率: {fold_accuracies}")
    print(f"平均验证集准确率: {np.mean(fold_accuracies):.2f}%")
    
    return best_models, fold_accuracies

# 更新后的主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 创建保存模型的目录
    os.makedirs('pytorch_models', exist_ok=True)
    
    print('='*60)
    
    # 1. 加载数据
    X = np.load('./fusai_train_features.npy', allow_pickle=True)
    Y = np.load('./fusai_train_labels.npy', allow_pickle=True)

    print(f"特征形状: {X.shape}, 标签形状: {Y.shape}")
    print(f"标签类别数: {len(np.unique(Y))}, 标签分布: {np.bincount(Y)}")
    
    # 6. 开启5折交叉验证
    num_classes = len(np.unique(Y))
    _, fold_accuracies = train_with_kfold(
        model_class=Conv1DModel,  # 替换成你希望使用的模型，如MLP或Conv1DModel
        features=X,
        labels=Y,
        num_classes=num_classes,
        num_epochs=200,
        device=device,
        k=5
    )
    os.makedirs('./logs/fold',exist_ok=True)
    utils.plot_fold_accuracies(
        fold_accuracies=fold_accuracies,
        save_dir="./logs/fold",
        fold_names=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    )
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()