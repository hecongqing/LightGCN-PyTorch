"""
LightGCN模型训练脚本
基于阿里移动推荐数据集训练LightGCN模型
包含模型训练、验证和评估功能
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

from alibaba_dataloader import AlibabaDataset
from lightgcn_model import LightGCN, BPRLoss, evaluate_model


class LightGCNTrainer:
    """
    LightGCN模型训练器
    
    提供完整的模型训练、验证和评估功能
    """
    
    def __init__(self, model: LightGCN, dataset: AlibabaDataset, 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 batch_size: int = 1024, device: str = 'cpu'):
        """
        初始化训练器
        
        Args:
            model: LightGCN模型
            dataset: 数据集
            learning_rate: 学习率
            weight_decay: 权重衰减
            batch_size: 批次大小
            device: 设备
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 损失函数
        self.criterion = BPRLoss(lambda_reg=weight_decay)
        
        # 训练历史
        self.train_losses = []
        self.val_metrics = []
        
        # 获取图数据
        self.edge_index = dataset.get_edge_index().to(device)
        self.train_data = dataset.get_train_data()
        self.test_data = dataset.get_test_data()
        self.user_item_dict = dataset.get_user_item_interactions()
        
        print(f"训练器初始化完成:")
        print(f"- 设备: {device}")
        print(f"- 批次大小: {batch_size}")
        print(f"- 学习率: {learning_rate}")
        print(f"- 权重衰减: {weight_decay}")
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        Returns:
            平均损失值
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 随机打乱训练数据
        indices = np.random.permutation(len(self.train_data))
        
        # 分批训练
        for start_idx in tqdm(range(0, len(indices), self.batch_size), 
                             desc="Training", leave=False):
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # 获取批次数据
            batch_users = []
            batch_pos_items = []
            batch_neg_items = []
            
            for idx in batch_indices:
                user_id, pos_item_id = self.train_data[idx]
                
                # 负采样
                neg_item_id = self.dataset.negative_sampling(user_id, 1)[0]
                
                batch_users.append(user_id)
                batch_pos_items.append(pos_item_id)
                batch_neg_items.append(neg_item_id)
            
            # 转换为张量
            user_ids = torch.LongTensor(batch_users).to(self.device)
            pos_item_ids = torch.LongTensor(batch_pos_items).to(self.device)
            neg_item_ids = torch.LongTensor(batch_neg_items).to(self.device)
            
            # 计算预测得分
            pos_scores = self.model.predict(user_ids, pos_item_ids, self.edge_index)
            neg_scores = self.model.predict(user_ids, neg_item_ids, self.edge_index)
            
            # 计算损失
            loss = self.criterion(pos_scores, neg_scores, self.model)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, k_list: List[int] = [10, 20]) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            k_list: 评估的K值列表
            
        Returns:
            评估指标字典
        """
        return evaluate_model(
            self.model, 
            self.test_data, 
            self.edge_index, 
            self.user_item_dict, 
            k_list, 
            self.device
        )
    
    def train(self, num_epochs: int = 100, eval_every: int = 10, 
              early_stopping_patience: int = 10, save_path: str = None) -> Dict:
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            eval_every: 评估间隔
            early_stopping_patience: 早停耐心值
            save_path: 模型保存路径
            
        Returns:
            训练历史
        """
        print(f"开始训练，共{num_epochs}轮...")
        
        best_recall = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            epoch_time = time.time() - start_time
            
            # 定期评估
            if (epoch + 1) % eval_every == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"训练损失: {train_loss:.4f}")
                print(f"训练时间: {epoch_time:.2f}秒")
                
                # 评估模型
                metrics = self.evaluate()
                self.val_metrics.append(metrics)
                
                print("评估结果:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                # 早停检查
                current_recall = metrics.get('Recall@20', 0)
                if current_recall > best_recall:
                    best_recall = current_recall
                    patience_counter = 0
                    
                    # 保存最佳模型
                    if save_path:
                        self.save_model(save_path)
                        print(f"模型已保存到: {save_path}")
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"早停触发，在第{epoch+1}轮停止训练")
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, 损失: {train_loss:.4f}, 时间: {epoch_time:.2f}秒")
        
        print("训练完成!")
        
        # 返回训练历史
        return {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'best_recall': best_recall
        }
    
    def save_model(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'num_users': self.model.num_users,
                'num_items': self.model.num_items,
                'embedding_dim': self.model.embedding_dim,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout
            },
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        torch.save(checkpoint, save_path)
    
    def load_model(self, load_path: str):
        """
        加载模型
        
        Args:
            load_path: 模型文件路径
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        print(f"模型已从 {load_path} 加载")
    
    def plot_training_curves(self, save_path: str = None):
        """
        绘制训练曲线
        
        Args:
            save_path: 图片保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失
        axes[0, 0].plot(self.train_losses)
        axes[0, 0].set_title('训练损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 提取验证指标
        if self.val_metrics:
            epochs = list(range(0, len(self.val_metrics) * 10, 10))
            
            # Recall曲线
            recall_10 = [m.get('Recall@10', 0) for m in self.val_metrics]
            recall_20 = [m.get('Recall@20', 0) for m in self.val_metrics]
            
            axes[0, 1].plot(epochs, recall_10, label='Recall@10', marker='o')
            axes[0, 1].plot(epochs, recall_20, label='Recall@20', marker='s')
            axes[0, 1].set_title('Recall')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Precision曲线
            precision_10 = [m.get('Precision@10', 0) for m in self.val_metrics]
            precision_20 = [m.get('Precision@20', 0) for m in self.val_metrics]
            
            axes[1, 0].plot(epochs, precision_10, label='Precision@10', marker='o')
            axes[1, 0].plot(epochs, precision_20, label='Precision@20', marker='s')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # NDCG曲线
            ndcg_10 = [m.get('NDCG@10', 0) for m in self.val_metrics]
            ndcg_20 = [m.get('NDCG@20', 0) for m in self.val_metrics]
            
            axes[1, 1].plot(epochs, ndcg_10, label='NDCG@10', marker='o')
            axes[1, 1].plot(epochs, ndcg_20, label='NDCG@20', marker='s')
            axes[1, 1].set_title('NDCG')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('NDCG')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        plt.show()


def main():
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    data_path = "./alibaba_data"
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = AlibabaDataset(
        data_path=data_path,
        behavior_types=[4],  # 只考虑购买行为
        test_size=0.2,
        random_state=42
    )
    
    # 模型参数
    model_config = {
        'num_users': dataset.num_users,
        'num_items': dataset.num_items,
        'embedding_dim': 64,
        'num_layers': 3,
        'dropout': 0.1
    }
    
    print(f"模型配置: {model_config}")
    
    # 创建模型
    model = LightGCN(**model_config)
    
    # 创建训练器
    trainer = LightGCNTrainer(
        model=model,
        dataset=dataset,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=1024,
        device=device
    )
    
    # 训练参数
    training_config = {
        'num_epochs': 200,
        'eval_every': 10,
        'early_stopping_patience': 15,
        'save_path': './models/lightgcn_alibaba.pth'
    }
    
    print(f"训练配置: {training_config}")
    
    # 开始训练
    history = trainer.train(**training_config)
    
    # 最终评估
    print("\n最终评估结果:")
    final_metrics = trainer.evaluate(k_list=[5, 10, 20])
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 绘制训练曲线
    trainer.plot_training_curves('./results/training_curves.png')
    
    # 保存训练历史
    with open('./results/training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print("训练完成！")


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    main()