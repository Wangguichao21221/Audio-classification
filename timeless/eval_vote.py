import numpy as np
import torch
import os
import glob
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from train import AudioDataset
device = 'cpu'
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5,
              'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream': 11,
              'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17,
              'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}

# ====================== 3. 加载测试数据 ======================
# 加载测试特征和文件名
test_X = np.load('./fusai_test_mfcc_128.npy', allow_pickle=True)
files = np.load("./fusai_test_filenames.npy", allow_pickle=True)
print(f"测试样本数量: {len(test_X)}, 文件名数量: {len(files)}")

# 重塑为PyTorch格式: (batch, channels, length)（添加通道维度）
test_X = test_X[:, np.newaxis, :]

# 构建测试数据集和DataLoader（和原脚本一致）
test_dataset = AudioDataset(test_X, np.zeros(len(test_X))) 
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ====================== 4. 加载result文件夹下所有pth权重文件 ======================
# 获取result文件夹下所有.pth文件（递归查找，支持子文件夹）
pth_dir = "./result"
pth_files = glob.glob(os.path.join(pth_dir, "**/*.pth"), recursive=True)

# 过滤空文件/无效文件
pth_files = [f for f in pth_files if os.path.getsize(f) > 0]
if not pth_files:
    raise ValueError(f"在 {pth_dir} 文件夹下未找到任何.pth权重文件！")
print(f"找到 {len(pth_files)} 个权重文件: {pth_files}")

# ====================== 5. 定义模型结构（和train.py中的Conv1DModel一致） ======================
# 从train.py导入模型（若导入失败，需手动复制Conv1DModel的定义）
from train import Conv1DModel

# ====================== 6. 多模型预测 + 投票逻辑 ======================
# 存储所有模型的预测结果：shape = [模型数量, 测试样本数量]
all_predictions = []

# 遍历每个权重文件，加载模型并预测
for idx, pth_file in enumerate(pth_files):
    print(f"\n正在加载第 {idx+1}/{len(pth_files)} 个模型: {pth_file}")
    
    # 初始化模型
    model = Conv1DModel(input_dim=128, num_classes=20)
    
    # 加载权重（兼容仅存储state_dict和存储checkpoint字典的情况）
    checkpoint = torch.load(pth_file, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 原脚本的checkpoint格式（包含model_state_dict）
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 仅存储model.state_dict()的情况
        model.load_state_dict(checkpoint)
    
    # 模型设为评估模式
    model.eval()
    
    # 单模型预测
    model_preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            # 存储数字标签（便于后续投票）
            model_preds.append(predicted.item())
    
    # 将当前模型的预测结果加入总列表
    all_predictions.append(model_preds)
    print(f"模型 {pth_file} 预测完成，共预测 {len(model_preds)} 个样本")

# 转置预测结果：shape = [测试样本数量, 模型数量] → 每个样本对应所有模型的预测
all_predictions = np.array(all_predictions).T
print(f"\n所有模型预测完成，预测结果矩阵形状: {all_predictions.shape}")

# 投票逻辑：对每个样本，取所有模型预测中出现次数最多的标签
final_predictions = []
for sample_preds in all_predictions:
    # 统计每个标签的出现次数
    counter = Counter(sample_preds)
    # 取出现次数最多的标签（若平局，取第一个出现的）
    most_common_label = counter.most_common(1)[0][0]
    # 转换为文字标签
    final_predictions.append(label_dict_inv[most_common_label])

# ====================== 7. 输出最终CSV文件 ======================
output_file = "submission_vote.csv"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("name,label\n")
    for filename, label in zip(files, final_predictions):
        f.write(f'{filename},{label}\n')

# 打印统计信息
print(f"\n========== 投票完成 ==========")
print(f"参与投票的模型数量: {len(pth_files)}")
print(f"总共预测 {len(final_predictions)} 个样本")
print(f"最终结果已保存至: {output_file}")

# 可选：打印投票结果统计（前10个样本）
print("\n前10个样本的投票详情：")
for i in range(min(10, len(files))):
    sample_preds = all_predictions[i]
    pred_labels = [label_dict_inv[p] for p in sample_preds]
    print(f"样本 {files[i]}: 各模型预测 = {pred_labels} → 最终投票结果 = {final_predictions[i]}")