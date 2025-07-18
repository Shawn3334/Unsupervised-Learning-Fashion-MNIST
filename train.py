import torch
import torch.nn as nn
import torchvision
import time
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from model import AutoEncoder

# 超参数配置
class Config:
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 64
    input_dim = 28 * 28
    hidden_dim = 128
    latent_dim = 10
    
    def to_dict(self):
        """将配置转换为字典，方便保存"""
        return {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim
        }

def create_experiment_dir():
    """创建实验目录"""
    # 确保run目录存在
    if not os.path.exists("run"):
        os.makedirs("run")
    
    # 查找现有的实验目录
    existing_exps = [d for d in os.listdir("run") if d.startswith("exp")]
    
    # 确定新的实验编号
    if not existing_exps:
        new_exp_num = 1
    else:
        # 提取现有实验的编号
        exp_nums = [int(exp.replace("exp", "")) for exp in existing_exps if exp.replace("exp", "").isdigit()]
        new_exp_num = max(exp_nums) + 1 if exp_nums else 1
    
    # 创建新的实验目录
    exp_dir = os.path.join("run", f"exp{new_exp_num}")
    os.makedirs(exp_dir)
    
    # 创建子目录
    os.makedirs(os.path.join(exp_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    
    return exp_dir

def load_data(batch_size):
    """加载Fashion-MNIST数据集"""
    # 训练集
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./datasets/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    
    # 测试集
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./datasets/',
        train=False,
        transform=torchvision.transforms.ToTensor()
    )
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset

def evaluate_model(model, test_loader, criterion, device, config):
    """评估模型在测试集上的性能"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _ in test_loader:
            # 确保批次大小正确
            if images.size(0) != config.batch_size:
                continue
                
            images = images.to(device).view(config.batch_size, -1)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    model.train()
    return avg_loss

def save_reconstruction_images(model, test_loader, device, epoch, exp_dir, num_samples=3):
    """保存重构图像对比"""
    model.eval()
    
    # 获取一批测试数据
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
    
    # 保存图像
    save_path = os.path.join(exp_dir, "images", f"reconstruction_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    
    model.train()

def train_model(model, train_loader, test_loader, config, device, exp_dir):
    """训练自编码器模型"""
    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    
    print("开始训练...")
    model.train()
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        # 使用tqdm创建进度条
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for i, (images, _) in progress_bar:
            # 将图像展平并移到设备上
            images = images.to(device).view(config.batch_size, -1)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 在测试集上评估模型
        val_loss = evaluate_model(model, test_loader, criterion, device, config)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{config.num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        accuracy_metrics = calculate_accuracy(model, test_loader, device, config)
        print(f"Reconstruction Similarity: {accuracy_metrics['reconstruction_similarity']:.4f}, "
            f"Clustering Accuracy: {accuracy_metrics['clustering_accuracy']:.4f}")
        
        # 保存重构图像
        save_reconstruction_images(model, test_loader, device, epoch+1, exp_dir)
        
        # 每个epoch保存一次模型
        model_save_path = os.path.join(exp_dir, "models", f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config.num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, config.num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "images", "loss_curve.png"))
    plt.close()
    
    # 保存损失数据
    loss_data = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open(os.path.join(exp_dir, "loss_data.json"), "w") as f:
        json.dump(loss_data, f)
    
    print("训练完成！")
    return model

def calculate_accuracy(model, test_loader, device, config):
    """计算模型的精度
    
    对于自编码器，我们可以通过以下方式评估精度：
    1. 重构精度：计算原始图像和重构图像的相似度
    2. 聚类精度：使用潜在表示进行聚类，计算与真实标签的匹配度
    """
    model.eval()
    total_samples = 0
    reconstruction_similarity = 0
    clustering_accuracy = 0
    
    # 收集所有潜在表示和标签
    all_latent = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            # 确保批次大小正确
            if images.size(0) != config.batch_size:
                continue
                
            images = images.to(device).view(config.batch_size, -1)
            
            # 获取潜在表示和重构图像
            latent = model.get_latent_rep(images)
            outputs = model(images)
            
            # 计算重构相似度（1 - 平均绝对误差）
            reconstruction_error = torch.abs(outputs - images).mean(dim=1)
            reconstruction_similarity += (1 - reconstruction_error).sum().item()
            
            # 收集潜在表示和标签用于聚类评估
            all_latent.append(latent.cpu())
            all_labels.append(labels)
            
            total_samples += images.size(0)
    
    # 计算平均重构相似度
    avg_reconstruction_similarity = reconstruction_similarity / total_samples
    
    # 计算聚类精度（可选，需要额外的聚类算法）
    if len(all_latent) > 0:
        all_latent = torch.cat(all_latent, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # 使用K-means聚类
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        kmeans = KMeans(n_clusters=10, random_state=0).fit(all_latent)
        cluster_labels = kmeans.labels_
        
        # 计算调整兰德指数（衡量聚类与真实标签的匹配度）
        clustering_accuracy = adjusted_rand_score(all_labels, cluster_labels)
    
    model.train()
    return {
        "reconstruction_similarity": avg_reconstruction_similarity,
        "clustering_accuracy": clustering_accuracy
    }

def main():
    """主训练函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练Fashion-MNIST自编码器')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    args = parser.parse_args()
    
    # 创建实验目录
    exp_dir = create_experiment_dir()
    print(f"实验结果将保存在: {exp_dir}")
    
    # 加载配置
    config = Config()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        print("已从配置文件加载超参数")
    
    # 保存超参数
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=4)
    
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, test_loader, train_dataset, test_dataset = load_data(config.batch_size)
    
    # 创建模型
    model = AutoEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim
    ).to(device)
    
    # 训练模型
    trained_model = train_model(model, train_loader, test_loader, config, device, exp_dir)
    
    # 保存最终模型
    final_model_path = os.path.join(exp_dir, "models", "final_model.pth")
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"最终模型已保存为: {final_model_path}")
    
    return trained_model, test_loader, test_dataset, device, exp_dir

if __name__ == "__main__":
    main()