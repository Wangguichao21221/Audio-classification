import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Tuple

def plot_training_metrics_separate(
    train_losses_per_epoch, 
    test_losses_per_epoch,
    train_accuracy_per_epoch, 
    test_accuracy_per_epoch,
    save_dir="training_plots"  # 默认保存文件夹
):
    """
    可视化训练和测试过程中的损失与准确率变化，并将四个指标分别保存为独立图片
    
    参数:
    train_losses_per_epoch (list): 每个epoch的训练损失列表
    test_losses_per_epoch (list): 每个epoch的测试损失列表
    train_accuracy_per_epoch (list): 每个epoch的训练准确率列表
    test_accuracy_per_epoch (list): 每个epoch的测试准确率列表
    save_dir (str): 图片保存文件夹路径，默认为当前目录下的training_plots
    
    输出:
    生成四个独立的可视化图片并保存到指定文件夹
    """
    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
    # 验证输入数据长度一致
    epochs = len(train_losses_per_epoch)
    if len(test_losses_per_epoch) != epochs or len(train_accuracy_per_epoch) != epochs or len(test_accuracy_per_epoch) != epochs:
        raise ValueError("所有输入列表的长度必须相同！")
    
    # 创建保存文件夹（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成epoch轴（从1开始）
    epoch_range = np.arange(1, epochs + 1)
    
    # --------------------- 1. 绘制训练损失曲线 ---------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(epoch_range, train_losses_per_epoch, 'b-', linewidth=2, marker='o', markersize=4, label='train loss')
    ax1.set_title('Training loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(1, epochs)
    plt.tight_layout()
    # 保存图片
    fig1.savefig(os.path.join(save_dir, 'train_loss.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)  # 关闭画布释放内存
    
    # --------------------- 2. 绘制测试损失曲线 ---------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(epoch_range, test_losses_per_epoch, 'r-', linewidth=2, marker='s', markersize=4, label='val loss', color='red')
    ax2.set_title('validation loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(1, epochs)
    plt.tight_layout()
    # 保存图片
    fig2.savefig(os.path.join(save_dir, 'test_loss.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # --------------------- 3. 绘制训练准确率曲线 ---------------------
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(epoch_range, train_accuracy_per_epoch, 'g-', linewidth=2, marker='o', markersize=4, label='training accuracy', color='green')
    ax3.set_title('Training accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(1, epochs)
    ax3.set_ylim(0, 1.05)  # 准确率范围限制
    plt.tight_layout()
    # 保存图片
    fig3.savefig(os.path.join(save_dir, 'train_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # --------------------- 4. 绘制测试准确率曲线 ---------------------
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(epoch_range, test_accuracy_per_epoch, 'orange', linewidth=2, marker='s', markersize=4, label='val accuracy')
    ax4.set_title('Validation accuracy', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(1, epochs)
    ax4.set_ylim(0, 1.05)  # 准确率范围限制
    plt.tight_layout()
    # 保存图片
    fig4.savefig(os.path.join(save_dir, 'test_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print(f"四张图片已成功保存到文件夹：{os.path.abspath(save_dir)}")


def plot_training_metrics_complete(
    train_losses_per_epoch, 
    test_losses_per_epoch,
    train_accuracy_per_epoch, 
    test_accuracy_per_epoch,
    save_dir="training_plots"
):
    """
    完整版本：既保存四个独立图片，也保存组合图
    """
    # 先保存独立图片
    plot_training_metrics_separate(
        train_losses_per_epoch, test_losses_per_epoch,
        train_accuracy_per_epoch, test_accuracy_per_epoch,
        save_dir
    )
    
    # 再绘制并保存组合图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs = len(train_losses_per_epoch)
    epoch_range = np.arange(1, epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Training', fontsize=16, fontweight='bold')
    
    # 损失曲线
    ax1.plot(epoch_range, train_losses_per_epoch, 'b-', linewidth=1, marker='o', markersize=4, label='training loss')
    ax1.plot(epoch_range, test_losses_per_epoch, 'r-', linewidth=1, marker='s', markersize=4, label='val loss')
    ax1.set_title('Training and Validation loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(1, epochs)
    
    # 准确率曲线
    ax2.plot(epoch_range, train_accuracy_per_epoch, 'g-', linewidth=1, marker='o', markersize=4, label='training accuracy')
    ax2.plot(epoch_range, test_accuracy_per_epoch, 'orange', linewidth=1, marker='s', markersize=4, label='val accuracy')
    ax2.set_title('Training and Validation accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(1, epochs)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, 'training_metrics_combined.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"组合图已保存：{os.path.join(save_dir, 'training_metrics_combined.png')}")
def plot_fold_accuracies(fold_accuracies, 
                         save_dir="fold_accuracies_plots",
                         plot_type="all",  # all/line/bar/box
                         fold_names=None,
                         title="K折交叉验证准确率分布",
                         figsize=(10, 6),
                         dpi=300):
    """
    可视化K折交叉验证的折准确率，并保存为图片
    
    参数:
    fold_accuracies (list): 各折的准确率列表，如[0.85, 0.88, 0.90, 0.87, 0.89]
    save_dir (str): 图片保存文件夹路径，默认为当前目录下的fold_plots
    plot_type (str): 绘图类型，可选值：
                     - "all": 保存所有类型的图（折线+柱状+箱线）
                     - "line": 仅折线图
                     - "bar": 仅柱状图
                     - "box": 仅箱线图
    fold_names (list): 各折的名称列表，如["Fold 1", "Fold 2"]，默认自动生成
    title (str): 图表主标题
    figsize (tuple): 图片尺寸，(宽度, 高度)
    dpi (int): 图片分辨率
    
    返回:
    None: 图片保存到指定文件夹
    """
    # 输入验证
    if not isinstance(fold_accuracies, (list, np.ndarray)):
        raise TypeError("fold_accuracies必须是列表或numpy数组")
    if len(fold_accuracies) == 0:
        raise ValueError("fold_accuracies不能为空列表")
    if any(not isinstance(acc, (int, float)) for acc in fold_accuracies):
        raise ValueError("fold_accuracies中的所有元素必须是数字")
    
    # 转换为numpy数组方便计算
    acc_array = np.array(fold_accuracies)
    n_folds = len(acc_array)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
    # 创建保存文件夹
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成折名称
    if fold_names is None:
        fold_names = [f"Fold {i+1}" for i in range(n_folds)]
    else:
        if len(fold_names) != n_folds:
            raise ValueError(f"fold_names长度({len(fold_names)})必须与fold_accuracies长度({n_folds})一致")
    
    # 计算统计信息
    mean_acc = np.mean(acc_array)
    std_acc = np.std(acc_array)
    max_acc = np.max(acc_array)
    min_acc = np.min(acc_array)
    
    # --------------------- 1. 折线图 ---------------------
    if plot_type in ["all", "line"]:
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制折线
        ax.plot(fold_names, acc_array, 'b-', linewidth=2.5, marker='o', 
                markersize=8, markerfacecolor='lightblue', markeredgecolor='blue')
        
        # 绘制均值线
        ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_acc:.4f} (±{std_acc:.4f})')
        
        # 设置样式
        ax.set_title(f"{title}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Fold", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(max(0, min_acc - 0.05), min(1.05, max_acc + 0.05))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        
        # 添加数值标注
        for i, acc in enumerate(acc_array):
            ax.text(i, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 添加统计信息文本
        stats_text = f'Max: {max_acc:.4f}\nMin: {min_acc:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        # 保存图片
        line_path = os.path.join(save_dir, 'fold_accuracies_line.png')
        fig.savefig(line_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"折线图已保存: {os.path.abspath(line_path)}")
    
    # --------------------- 2. 柱状图 ---------------------
    if plot_type in ["all", "bar"]:
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制柱状图
        bars = ax.bar(fold_names, acc_array, width=0.6, 
                      color='skyblue', edgecolor='navy', linewidth=1.5)
        
        # 绘制均值线
        ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
                   label=f'mean: {mean_acc:.4f} (±{std_acc:.4f})')
        
        # 设置样式
        ax.set_title(f"{title}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Fold", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(max(0, min_acc - 0.05), min(1.05, max_acc + 0.05))
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(fontsize=11)
        
        # 添加数值标注
        for bar, acc in zip(bars, acc_array):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 添加统计信息文本
        stats_text = f'Max: {max_acc:.4f}\Min: {min_acc:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        # 保存图片
        bar_path = os.path.join(save_dir, 'fold_accuracies_bar.png')
        fig.savefig(bar_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"柱状图已保存: {os.path.abspath(bar_path)}")
    
    print(f"\n所有图片已保存到文件夹: {os.path.abspath(save_dir)}")
    print(f"统计信息:")
    print(f"  平均准确率: {mean_acc:.4f}")
    print(f"  标准差:     {std_acc:.4f}")
    print(f"  最高准确率: {max_acc:.4f}")
    print(f"  最低准确率: {min_acc:.4f}")
def plot_seed_accuracy(
    seed_accuracy: List[float],
    save_dir: str = "seed_accuracy_plots",
    seed_values: Optional[List[int]] = None,
    plot_type: str = "all",  # all/bar/box/violin/line
    title: str = "不同随机种子下的模型准确率",
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
    accuracy_label: str = "准确率"
) -> None:
    """
    可视化不同随机种子下的模型准确率，分析结果的稳定性
    
    参数:
    ----------
    seed_accuracy : List[float]
        不同随机种子对应的准确率列表，如 [0.85, 0.86, 0.84, 0.855]
    save_dir : str
        图片保存目录，默认创建 "seed_accuracy_plots" 文件夹
    seed_values : Optional[List[int]]
        对应的随机种子数值列表，如 [42, 100, 200, 300]，默认自动生成
    plot_type : str
        绘图类型：
        - "all": 保存所有类型图表（柱状图+箱线图+小提琴图+折线图）
        - "bar": 仅柱状图（对比各种子准确率）
        - "box": 仅箱线图（展示分布和离群值）
        - "violin": 仅小提琴图（展示概率分布）
        - "line": 仅折线图（展示种子变化趋势）
    title : str
        图表主标题
    figsize : Tuple[int, int]
        图片尺寸 (宽度, 高度)
    dpi : int
        图片分辨率（默认300高清）
    accuracy_label : str
        y轴标签（默认"准确率"）
    
    输出:
    ----------
    None
        生成可视化图片并保存到指定目录，控制台打印统计信息
    """
    # 输入验证
    if not isinstance(seed_accuracy, list) or len(seed_accuracy) == 0:
        raise ValueError("seed_accuracy必须是非空列表")
    if not all(isinstance(x, (int, float)) for x in seed_accuracy):
        raise ValueError("seed_accuracy中所有元素必须是数值类型")
    
    # 转换为numpy数组便于计算
    acc_array = np.array(seed_accuracy)
    n_seeds = len(acc_array)
    
    # 处理随机种子标签
    if seed_values is None:
        seed_values = [f"Seed {i+1}" for i in range(n_seeds)]
    else:
        if len(seed_values) != n_seeds:
            raise ValueError(f"seed_values长度({len(seed_values)})必须与seed_accuracy长度({n_seeds})一致")
        # 转换为字符串便于绘图
        seed_values = [str(x) for x in seed_values]
    
    # 计算关键统计指标（评估稳定性）
    mean_acc = np.mean(acc_array)
    std_acc = np.std(acc_array)
    cv_acc = (std_acc / mean_acc) * 100  # 变异系数（越小越稳定）
    max_acc = np.max(acc_array)
    min_acc = np.min(acc_array)
    range_acc = max_acc - min_acc  # 极差（越小越稳定）
    
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # --------------------- 1. 柱状图（对比各种子准确率） ---------------------
    if plot_type in ["all", "bar"]:
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制柱状图
        bars = ax.bar(
            seed_values, acc_array,
            width=0.6, color='#3498db', edgecolor='#2980b9', linewidth=1.2,
            alpha=0.8
        )
        
        # 添加均值线
        ax.axhline(
            y=mean_acc, color='#e74c3c', linestyle='--', linewidth=2,
            label=f'Mean: {mean_acc:.4f} (±{std_acc:.4f})'
        )
        
        # 数值标注
        for bar, acc in zip(bars, acc_array):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        
        # 样式配置
        ax.set_title(f"{title}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Random seed", fontsize=12)
        ax.set_ylabel(accuracy_label, fontsize=12)
        ax.set_ylim(max(0, min_acc - 0.02), min(1.02, max_acc + 0.02))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)
        
        # 添加稳定性评估文本
        
        plt.tight_layout()
        bar_path = os.path.join(save_dir, 'seed_accuracy_bar.png')
        fig.savefig(bar_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 柱状图已保存: {os.path.abspath(bar_path)}")
    
    # --------------------- 4. 折线图（展示种子变化趋势） ---------------------
    if plot_type in ["all", "line"]:
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制折线图
        ax.plot(
            seed_values, acc_array,
            color='#3498db', linewidth=2.5, marker='o',
            markersize=8, markerfacecolor='#2ecc71', markeredgecolor='#27ae60'
        )
        
        # 添加均值带（均值±标准差）
        ax.fill_between(
            seed_values, mean_acc - std_acc, mean_acc + std_acc,
            color='#f39c12', alpha=0.2, label=f'mean±std_var (±{std_acc:.4f})'
        )
        
        # 数值标注
        for i, acc in enumerate(acc_array):
            ax.text(
                i, acc + 0.003, f'{acc:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        
        # 样式配置
        ax.set_title(f"{title}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Random seeds", fontsize=12)
        ax.set_ylabel(accuracy_label, fontsize=12)
        ax.set_ylim(max(0, min_acc - 0.02), min(1.02, max_acc + 0.02))
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        line_path = os.path.join(save_dir, 'seed_accuracy_line.png')
        fig.savefig(line_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 折线图已保存: {os.path.abspath(line_path)}")
    
    # 打印统计总结
    print("\n📊 随机种子准确率统计总结:")
    print(f"   种子数量: {n_seeds}")
    print(f"   平均准确率: {mean_acc:.4f}")
    print(f"   标准差: {std_acc:.4f}")
    print(f"   变异系数: {cv_acc:.2f}% (越小越稳定)")
    print(f"   准确率范围: [{min_acc:.4f}, {max_acc:.4f}] (极差: {range_acc:.4f})")
    print(f"   中位数准确率: {np.median(acc_array):.4f}")
    print(f"\n💾 所有图片已保存至: {os.path.abspath(save_dir)}")

def plot_model_accuracy_simple(model_accuracies, save_path="model_accuracy.png"):
    """
    极简版多模型准确率柱状图可视化
    
    参数:
    model_accuracies (dict): 模型名称-准确率字典，如 {"ModelA":0.85, "ModelB":0.88}
    save_path (str): 图片保存路径（含文件名）
    """
    # 基础配置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取数据
    models = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())
    
    # 创建画布并绘制柱状图
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies)
    for bar in bars:
        height = bar.get_height()
        # 在柱子顶部居中位置标注数值，保留两位小数
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',  # 数值格式
                 ha='center', va='bottom',  # 水平居中、垂直在柱子上方
                 fontsize=10)  # 字体大小
    # 基础标签
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Different Model Comparision')
    
    # 自动调整布局并保存
    plt.tight_layout()
    # 自动创建保存目录（如果路径包含文件夹）
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"柱状图已保存至: {os.path.abspath(save_path)}")


def plot_multi_model_train_acc(
    model_acc_dict, 
    save_path="multi_model_train_acc.png",
    epoch_label="训练轮次",
    acc_label="准确率"
):
    """
    多模型训练准确率变化折线图（同图展示）
    
    参数:
    model_acc_dict (dict): 模型名称-准确率列表字典，如 {"模型1":[0.7,0.8,0.85], "模型2":[0.65,0.78,0.83]}
    save_path (str): 图片保存路径
    epoch_label (str): x轴标签
    acc_label (str): y轴标签
    """
    # 中文支持配置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 定义折线样式（颜色+标记，保证不同模型区分明显）
    styles = ['-o', '-s', '-^', '-D', '-*']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 遍历每个模型绘制折线
    for idx, (model_name, acc_list) in enumerate(model_acc_dict.items()):
        # 生成轮次（从1开始）
        epochs = list(range(1, len(acc_list)+1))
        # 绘制折线：循环使用样式/颜色，保证不重复
        style = styles[idx % len(styles)]
        color = colors[idx % len(colors)]
        plt.plot(epochs, acc_list, style, label=model_name, color=color, linewidth=1)
    
    # 基础标签与图例
    plt.xlabel(epoch_label, fontsize=12)
    plt.ylabel(acc_label, fontsize=12)
    plt.title('多模型训练准确率变化对比', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)  # 图例放在右下角
    plt.grid(alpha=0.3, linestyle='--')  # 轻微网格辅助查看
    
    # 自动创建保存目录+保存图片
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"多模型准确率折线图已保存至: {os.path.abspath(save_path)}")

