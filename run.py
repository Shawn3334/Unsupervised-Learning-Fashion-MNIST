import os
import argparse
import subprocess
import sys
import json

def main():
    parser = argparse.ArgumentParser(description='运行Fashion-MNIST自编码器实验')
    # 基本操作参数
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--exp_dir', type=str, default=None, help='指定要可视化的实验目录')
    
    # 超参数设置
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--latent_dim', type=int, default=10, help='潜在空间维度')
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认执行训练和可视化
    if not (args.train or args.visualize):
        args.train = True
        args.visualize = True
    
    # 执行训练
    if args.train:
        print("开始训练模型...")
        # 创建临时配置文件
        config = {
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "input_dim": 28 * 28  # 这个是固定的，因为是MNIST数据集
        }
        
        # 将配置写入临时文件
        temp_config_path = "temp_config.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f)
        
        # 调用train.py并传递配置文件路径
        subprocess.run([sys.executable, "train.py", "--config", temp_config_path])
        
        # 训练完成后删除临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    # 执行可视化
    if args.visualize:
        print("开始可视化结果...")
        if args.exp_dir:
            subprocess.run([sys.executable, "visualize.py", "--exp_dir", args.exp_dir])
        else:
            subprocess.run([sys.executable, "visualize.py"])

if __name__ == "__main__":
    main()