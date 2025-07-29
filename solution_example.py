"""
阿里巴巴数据集loss不动问题的完整解决方案
展示修复后的训练过程
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from fixed_alibaba_dataloader import FixedAlibabaDataset, create_larger_sample_data
from lightgcn_model import LightGCN, BPRLoss
import time
from tqdm import tqdm

class ImprovedLightGCNTrainer:
    """改进的LightGCN训练器，解决loss不动问题"""
    
    def __init__(self, model, dataset, learning_rate=0.001, weight_decay=1e-5, 
                 batch_size=1024, device='cpu'):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        
        # 使用更小的权重衰减
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 使用更小的正则化系数
        self.criterion = BPRLoss(lambda_reg=weight_decay)
        
        # 训练历史
        self.train_losses = []
        self.bpr_losses = []
        self.reg_losses = []
        
        # 获取图数据
        self.edge_index = dataset.get_edge_index().to(device)
        self.train_data = dataset.get_train_data()
        self.test_data = dataset.get_test_data()
        self.user_item_dict = dataset.get_user_item_interactions()
        
        print(f"✅ 训练器初始化完成:")
        print(f"- 设备: {device}")
        print(f"- 批次大小: {batch_size}")
        print(f"- 学习率: {learning_rate}")
        print(f"- 权重衰减: {weight_decay}")
        print(f"- 训练样本数: {len(self.train_data)}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_bpr_loss = 0
        total_reg_loss = 0
        num_batches = 0
        
        # 随机打乱训练数据
        indices = np.random.permutation(len(self.train_data))
        
        # 分批训练
        pbar = tqdm(range(0, len(indices), self.batch_size), desc="Training")
        
        for start_idx in pbar:
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # 获取批次数据
            batch_users = []
            batch_pos_items = []
            batch_neg_items = []
            
            for idx in batch_indices:
                user_id, pos_item_id = self.train_data[idx]
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
            
            # 单独计算BPR损失和正则化损失用于监控
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            reg_loss = sum(torch.norm(param, p=2) ** 2 for param in self.model.parameters())
            reg_loss = self.criterion.lambda_reg * reg_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'BPR': f'{bpr_loss.item():.4f}',
                'Reg': f'{reg_loss.item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_bpr_loss = total_bpr_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        
        return avg_loss, avg_bpr_loss, avg_reg_loss
    
    def train(self, num_epochs=50, print_every=5):
        """训练模型"""
        print(f"\n🚀 开始训练，共{num_epochs}轮...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss, bpr_loss, reg_loss = self.train_epoch()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.bpr_losses.append(bpr_loss)
            self.reg_losses.append(reg_loss)
            
            epoch_time = time.time() - start_time
            
            # 定期打印进度
            if (epoch + 1) % print_every == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"总损失: {train_loss:.6f}")
                print(f"BPR损失: {bpr_loss:.6f}")
                print(f"正则化损失: {reg_loss:.6f}")
                print(f"训练时间: {epoch_time:.2f}秒")
                
                # 检查损失变化
                if epoch >= 10:
                    recent_losses = self.train_losses[-10:]
                    loss_change = abs(recent_losses[-1] - recent_losses[0])
                    if loss_change < 1e-6:
                        print("⚠️  损失变化很小，可能需要调整学习率")
                    else:
                        print(f"✅ 损失变化: {loss_change:.6f}")
        
        print("\n🎉 训练完成!")
        return {
            'train_losses': self.train_losses,
            'bpr_losses': self.bpr_losses,
            'reg_losses': self.reg_losses
        }
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 总损失
        axes[0].plot(epochs, self.train_losses, 'b-', linewidth=2)
        axes[0].set_title('总损失 (Total Loss)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # BPR损失
        axes[1].plot(epochs, self.bpr_losses, 'r-', linewidth=2)
        axes[1].set_title('BPR损失')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('BPR Loss')
        axes[1].grid(True, alpha=0.3)
        
        # 正则化损失
        axes[2].plot(epochs, self.reg_losses, 'g-', linewidth=2)
        axes[2].set_title('正则化损失')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Regularization Loss')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        plt.show()


def test_different_configurations():
    """测试不同的配置以找到最佳设置"""
    print("🧪 测试不同的配置以解决loss不动问题...")
    
    # 创建测试数据
    data_path = "./solution_test_data"
    create_larger_sample_data(data_path)
    
    # 配置组合
    configs = [
        {"name": "配置1: 原始设置", "lr": 0.001, "wd": 1e-4, "min_inter": 5},
        {"name": "配置2: 降低正则化", "lr": 0.001, "wd": 1e-5, "min_inter": 5},
        {"name": "配置3: 提高学习率", "lr": 0.01, "wd": 1e-5, "min_inter": 5},
        {"name": "配置4: 降低过滤条件", "lr": 0.001, "wd": 1e-5, "min_inter": 2},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"🔧 测试 {config['name']}")
        print(f"{'='*60}")
        
        try:
            # 加载数据集
            dataset = FixedAlibabaDataset(
                data_path=data_path,
                behavior_types=[4],
                min_user_interactions=config['min_inter'],
                min_item_interactions=config['min_inter'],
                test_size=0.2
            )
            
            # 创建模型
            model = LightGCN(
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                embedding_dim=64,
                num_layers=3,
                dropout=0.1
            )
            
            # 创建训练器
            trainer = ImprovedLightGCNTrainer(
                model=model,
                dataset=dataset,
                learning_rate=config['lr'],
                weight_decay=config['wd'],
                batch_size=512,
                device='cpu'
            )
            
            # 训练模型
            history = trainer.train(num_epochs=20, print_every=5)
            
            # 分析结果
            final_loss = history['train_losses'][-1]
            loss_reduction = history['train_losses'][0] - final_loss
            loss_stable = abs(history['train_losses'][-1] - history['train_losses'][-5]) < 1e-6
            
            results[config['name']] = {
                'final_loss': final_loss,
                'loss_reduction': loss_reduction,
                'loss_stable': loss_stable,
                'data_stats': dataset.get_data_statistics()
            }
            
            print(f"✅ {config['name']} 完成")
            print(f"   最终损失: {final_loss:.6f}")
            print(f"   损失下降: {loss_reduction:.6f}")
            print(f"   是否稳定: {'是' if not loss_stable else '否'}")
            
        except Exception as e:
            print(f"❌ {config['name']} 失败: {e}")
            results[config['name']] = {'error': str(e)}
    
    # 总结结果
    print(f"\n{'='*60}")
    print("📊 配置测试结果总结")
    print(f"{'='*60}")
    
    for name, result in results.items():
        print(f"\n{name}:")
        if 'error' in result:
            print(f"  ❌ 错误: {result['error']}")
        else:
            print(f"  最终损失: {result['final_loss']:.6f}")
            print(f"  损失下降: {result['loss_reduction']:.6f}")
            print(f"  数据统计: 用户{result['data_stats']['num_users']}, 商品{result['data_stats']['num_items']}, 交互{result['data_stats']['num_interactions']}")
    
    return results


def main():
    """主函数：展示完整的解决方案"""
    print("🎯 阿里巴巴数据集Loss不动问题的完整解决方案")
    print("="*60)
    
    # 1. 问题诊断总结
    print("\n📋 问题诊断总结:")
    print("主要问题: 数据预处理过于严格，导致训练数据不足")
    print("具体原因:")
    print("1. 最小交互次数设置为5，过滤掉了大量数据")
    print("2. 只保留购买行为(类型4)，数据量进一步减少")
    print("3. 商品子集过滤又减少了数据")
    print("4. 最终导致训练数据为空或极少")
    
    # 2. 解决方案
    print("\n🔧 解决方案:")
    print("1. 降低最小交互次数要求 (5 → 2)")
    print("2. 减少正则化强度 (1e-4 → 1e-5)")
    print("3. 调整学习率和优化器设置")
    print("4. 增加梯度裁剪防止梯度爆炸")
    print("5. 改进负采样方法")
    
    # 3. 运行测试
    results = test_different_configurations()
    
    # 4. 最佳实践建议
    print(f"\n💡 最佳实践建议:")
    print("1. 数据预处理:")
    print("   - 用户最小交互次数: 2-3")
    print("   - 商品最小交互次数: 2-3")
    print("   - 保留更多行为类型 [3,4] 而不是只有 [4]")
    print()
    print("2. 模型参数:")
    print("   - 学习率: 0.001-0.01")
    print("   - 权重衰减: 1e-5 到 1e-6")
    print("   - 批次大小: 512-2048")
    print("   - 嵌入维度: 64-128")
    print()
    print("3. 训练策略:")
    print("   - 使用梯度裁剪")
    print("   - 监控BPR损失和正则化损失")
    print("   - 定期检查损失变化")
    print("   - 使用学习率调度器")
    
    print(f"\n✅ 解决方案演示完成！")


if __name__ == "__main__":
    main()