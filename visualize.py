import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import torch
import numpy as np
import os
import json
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
import time
from model import AutoEncoder
from train import Config, load_data

def plot_sample_images(test_loader, exp_dir, num_samples=6):
    """绘制样本图像"""
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    print(f"数据形状: {example_data.shape}")
    
    fig = plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(example_data[i][0], cmap='gray', interpolation='none')
        ax.set_title(f'Ground Truth: {example_targets[i]}')
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "images", "sample_images.png"))
    plt.close()  # 关闭图形而不是显示

def plot_reconstructions(model, test_loader, device, exp_dir, num_samples=3):
    """绘制重构图像对比"""
    model.eval()
    
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    fig = plt.figure(figsize=(15, 10))
    
    for i in range(num_samples):
        # 原始图像
        ax = fig.add_subplot(2, num_samples, i + 1)
        ax.imshow(example_data[i][0], cmap='gray', interpolation='none')
        ax.set_title(f"Original: {example_targets[i]}")
        ax.set_axis_off()
        
        # 重构图像
        ax = fig.add_subplot(2, num_samples, i + num_samples + 1)
        with torch.no_grad():
            recon_img = model(example_data[i][0].view(1, -1).to(device))
            recon_img = recon_img.data.cpu().numpy().reshape(28, 28)
        ax.imshow(recon_img, cmap='gray')
        ax.set_title(f"Reconstructed: {example_targets[i]}")
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "images", "final_reconstructions.png"))
    plt.close()  # 关闭图形而不是显示

def compare_dimensionality_reduction(model, test_dataset, device, exp_dir, n_points=500):
    """比较不同的降维方法"""
    # 准备数据
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=n_points,
        shuffle=False
    )
    
    X, labels = next(iter(test_loader))
    
    # 获取潜在表示
    model.eval()
    with torch.no_grad():
        latent_X = model.get_latent_rep(X.to(device).view(n_points, -1))
        latent_X = latent_X.data.cpu().numpy()
    
    labels = labels.data.cpu().numpy()
    
    # 创建图形
    fig = plt.figure(figsize=(20, 5))
    
    # PCA
    print("执行PCA降维...")
    t0 = time.time()
    x_pca = PCA(n_components=2).fit_transform(latent_X)
    t1 = time.time()
    print(f"PCA时间: {t1 - t0:.2f} 秒")
    
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax.set_title('PCA')
    
    # Kernel PCA
    print("执行Kernel PCA降维...")
    t0 = time.time()
    x_kpca = KernelPCA(n_components=2, kernel='rbf').fit_transform(latent_X)
    t1 = time.time()
    print(f"Kernel PCA时间: {t1 - t0:.2f} 秒")
    
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(x_kpca[:, 0], x_kpca[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax.set_title('Kernel PCA')
    
    # t-SNE
    print("执行t-SNE降维...")
    t0 = time.time()
    x_tsne = TSNE(n_components=2).fit_transform(latent_X)
    t1 = time.time()
    print(f"t-SNE时间: {t1 - t0:.2f} 秒")
    
    ax = fig.add_subplot(1, 3, 3)
    scatter = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax.set_title('t-SNE')
    
    # 添加颜色条
    bounds = np.linspace(0, 10, 11)
    cb = plt.colorbar(scatter, spacing='proportional', ticks=bounds)
    cb.set_label('Class Colors')
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "images", "dimensionality_reduction.png"))
    plt.close()  # 关闭图形而不是显示

def plot_loss_curve(exp_dir):
    """绘制损失曲线"""
    # 加载损失数据
    with open(os.path.join(exp_dir, "loss_data.json"), "r") as f:
        loss_data = json.load(f)
    
    train_losses = loss_data["train_losses"]
    val_losses = loss_data["val_losses"]
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "images", "loss_curve_visualization.png"))
    plt.close()  # 关闭图形而不是显示

def main():
    """主可视化函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Visualize Fashion-MNIST autoencoder results')
    parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory to visualize')
    args = parser.parse_args()
    
    # 如果没有指定实验目录，使用最新的实验
    if args.exp_dir is None:
        if not os.path.exists("run"):
            print("未找到run目录，请先运行train.py")
            return
        
        existing_exps = [d for d in os.listdir("run") if d.startswith("exp")]
        if not existing_exps:
            print("未找到任何实验目录，请先运行train.py")
            return
        
        # 提取实验编号并找到最新的
        exp_nums = [int(exp.replace("exp", "")) for exp in existing_exps if exp.replace("exp", "").isdigit()]
        latest_exp_num = max(exp_nums) if exp_nums else 1
        exp_dir = os.path.join("run", f"exp{latest_exp_num}")
    else:
        exp_dir = args.exp_dir
    
    print(f"正在可视化实验: {exp_dir}")
    
    # 加载配置
    try:
        with open(os.path.join(exp_dir, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
    except FileNotFoundError:
        print(f"未找到配置文件: {os.path.join(exp_dir, 'config.json')}")
        return
    
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    _, test_loader, _, test_dataset = load_data(config.batch_size)
    
    # 加载训练好的模型
    model_path = os.path.join(exp_dir, "models", "final_model.pth")
    if not os.path.exists(model_path):
        print(f"未找到模型文件: {model_path}")
        return
    
    model = AutoEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    print("成功加载预训练模型")
    
    # 可视化
    print("\n1. 保存样本图像...")
    plot_sample_images(test_loader, exp_dir)
    
    print("\n2. 保存损失曲线...")
    plot_loss_curve(exp_dir)
    
    print("\n3. 保存重构对比...")
    plot_reconstructions(model, test_loader, device, exp_dir)
    
    print("\n4. 保存降维方法比较...")
    compare_dimensionality_reduction(model, test_dataset, device, exp_dir)
    
    print(f"\n所有可视化结果已保存到: {os.path.join(exp_dir, 'images')}")

if __name__ == "__main__":
    main()