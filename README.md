# 无监督Fashion-MNIST自编码器项目

## 🚀 项目简介

这是一个基于PyTorch实现的无监督学习项目，使用自编码器（AutoEncoder）对Fashion-MNIST数据集进行降维、特征学习和图像重构。项目通过自编码器学习服装图像的潜在表示，并提供了多种可视化方法来评估模型性能。

## 📂 项目结构
无监督fashion-MINST/
├── model.py # 自编码器模型定义
├── train.py # 模型训练和评估代码
├── run.py # 项目运行入口
├── visualize.py # 结果可视化工具
├── datasets/ # 数据集存储目录
└── run/
    └── exp*/ # 实验目录
        ├── config.json # 实验配置
        ├── loss_data.json # 损失数据
        ├── images/ # 可视化图像
        └── models/ # 保存的模型

## ✨ 功能特点

- **自编码器模型**：实现了基本的自编码器架构，包含编码器和解码器。
- **无监督学习**：不依赖标签信息进行特征学习。
- **多种评估指标**：
  - **重构相似度**：评估模型重构原始图像的能力。
  - **聚类准确率**：评估学习到的潜在表示的语义信息。
- **丰富的可视化**：
  - 原始与重构图像对比。
  - 训练和验证损失曲线。
  - 多种降维方法（PCA、Kernel PCA、t-SNE）的比较。
- **实验管理**：自动创建和管理实验目录，保存配置和结果。

## 🛠️ 安装依赖

```bash
pip install torch torchvision matplotlib numpy scikit-learn tqdm
```

## ⚙️ 使用方法

#### 1. 训练模型

```bash
python run.py --train
```

#### 2. 自定义训练参数

```bash
python run.py --train --num_epochs 50 --learning_rate 0.001 --batch_size 128 --hidden_dim 256 --latent_dim 20
```

#### 3. 可视化结果

可视化最新的实验结果：
```bash
python run.py --visualize
```

指定实验目录进行可视化：
```bash
python run.py --visualize --exp_dir run/exp1
```

#### 4. 同时训练和可视化（默认行为）

```bash
python run.py
```

## 📊 模型评估

模型通过以下两个指标进行评估：

- **重构相似度 (Reconstruction Similarity)**：计算为 `1 - 平均绝对误差`，值越高表示重构质量越好。
- **聚类准确率 (Clustering Accuracy)**：使用K-means对潜在表示进行聚类，并通过调整兰德指数（Adjusted Rand Index）与真实标签比较，评估潜在空间的语义信息。

## 🖼️ 可视化结果

训练后，可以在 `run/exp*/images/` 目录下查看以下可视化结果：

- `sample_images.png`：原始样本图像。
- `reconstruction_epoch_*.png`：每个训练周期的重构图像。
- `final_reconstructions.png`：最终模型的重构效果。
- `loss_curve.png` 和 `loss_curve_visualization.png`：训练和验证损失曲线。
- `dimensionality_reduction.png`：PCA、Kernel PCA和t-SNE降维结果比较。

## 💡 改进方向

- [ ] 增加网络深度和宽度（调整`hidden_dim`和`latent_dim`）。
- [ ] 实现变分自编码器（VAE）。
- [ ] 添加卷积层（CNN AutoEncoder）处理图像数据。
- [ ] 引入对比损失函数（如SimCLR）。
- [ ] 增加数据增强。
- [ ] 添加正则化和学习率调度。

## 📄 许可证

[MIT](LICENSE)