import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
from model import Conv1DModel,MLP
import random
# 定义数据集类
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 训练函数
def train_model(model, train_loader, test_loader, output_dir,seed,lr =1e-4,num_epochs=200, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    best_accuracy = 0.0
    best_model_path = None
    
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
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # 验证阶段
        model.eval()
        
        val_accuracy,val_loss = test_acc(model,device,test_loader)
        
        train_accuracy = 100.0 * train_correct / train_total
        
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
    new_path = os.path.join(output_dir,f"seed_{seed}_{best_accuracy:.4f}.pth")
    os.rename(best_model_path, new_path)
    return best_accuracy, new_path
def set_seed(seed: int):
    """
    设置全局随机种子，保证实验可复现
    
    Args:
        seed: 随机种子值（如0, 1, 42, 100等，不同值对应不同随机序列）
    """
    # 1. Python内置随机数
    random.seed(seed)
    
    # 2. NumPy随机数（若用到NumPy才需要）
    np.random.seed(seed)
    
    # 3. PyTorch CPU随机数
    torch.manual_seed(seed)
    
    # 4. PyTorch CUDA随机数（单GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时需加这行
    
    # 5. 禁用CuDNN的非确定性算法（保证GPU计算可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化（牺牲速度换确定性）
def rmdir_content(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # 1. 删除所有文件
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"删除文件失败 {file_path}：{e}")

# 主训练循环
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    save_model_path = 'pytorch_models'
    saved_best_model_path = "./result"
    # 创建保存模型的目录
    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(saved_best_model_path, exist_ok=True)
    
    # 1. 加载数据
    X = np.load('./fusai_train_features.npy', allow_pickle=True)
    Y = np.load('./fusai_train_labels.npy', allow_pickle=True)

    print(f"特征形状: {X.shape}, 标签形状: {Y.shape}")
    print(f"标签类别数: {len(np.unique(Y))}, 标签分布: {np.bincount(Y)}")
    for seed in range(124,135):
        set_seed(seed=seed)
        rmdir_content(save_model_path)
        
        print('='*30,f"seed:{seed}","="*30)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, 
            test_size=0.1,
            stratify=Y,
            random_state=seed
        )
        
        # 4. 重塑数据形状以适应Conv1D
        # PyTorch期望的形状: (batch_size, channels, length)
        X_train = X_train.reshape(-1, 1, X.shape[1])  # (900, 1, 128)
        X_test = X_test.reshape(-1, 1, X.shape[1])    # (100, 1, 128)
        
        print(f"训练集: {X_train.shape}, {Y_train.shape}")
        print(f"测试集: {X_test.shape}, {Y_test.shape}")
        
        # 5. 创建数据加载器
        train_dataset = AudioDataset(X_train, Y_train)
        test_dataset = AudioDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # 6. 创建模型
        num_classes = len(np.unique(Y))
        model = Conv1DModel(input_dim=X.shape[1], num_classes=num_classes)
        print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")
        
        # 7. 训练模型
        best_accuracy, best_model_path = train_model(
            model, train_loader, test_loader, 
            num_epochs=150, device=device,output_dir=saved_best_model_path,seed= seed
        )
        
        # 8. 重命名保存的模型文件
        
        # 9. 加载并测试最佳模型
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 在测试集上最终评
        
        test_accuracy ,loss= test_acc(model,device,test_loader)
        print(f"最终测试准确率: {test_accuracy:.4f}%")
        print(f"最佳模型保存为: {best_model_path}")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    files = glob.glob(os.path.join(save_model_path, "*"))
    for f in files:
      if os.path.isfile(f):
          os.remove(f)
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
            total_loss += loss.item()
    test_accuracy = 100.0 * test_correct / test_total
    return test_accuracy,total_loss
if __name__ == "__main__":
    main()