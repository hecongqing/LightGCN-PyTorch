"""
阿里移动推荐系统主程序
基于LightGCN模型的完整推荐系统实现

功能包括:
1. 数据加载和预处理
2. 模型训练
3. 推荐生成
4. 结果分析和可视化
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple
import json

from alibaba_dataloader import AlibabaDataset, create_sample_data
from lightgcn_model import LightGCN, evaluate_model
from train import LightGCNTrainer


class AlibabaRecommendationSystem:
    """
    阿里移动推荐系统
    
    完整的推荐系统实现，包含数据处理、模型训练、推荐生成等功能
    """
    
    def __init__(self, data_path: str, model_config: Dict = None, device: str = 'cpu'):
        """
        初始化推荐系统
        
        Args:
            data_path: 数据路径
            model_config: 模型配置
            device: 计算设备
        """
        self.data_path = data_path
        self.device = device
        self.model_config = model_config or {
            'embedding_dim': 64,
            'num_layers': 3,
            'dropout': 0.1
        }
        
        # 初始化组件
        self.dataset = None
        self.model = None
        self.trainer = None
        
        print(f"推荐系统初始化完成，使用设备: {device}")
    
    def load_data(self, behavior_types: List[int] = [4], test_size: float = 0.2):
        """
        加载数据集
        
        Args:
            behavior_types: 考虑的行为类型
            test_size: 测试集比例
        """
        print("正在加载数据集...")
        
        # 检查数据是否存在，如果不存在则创建示例数据
        if not os.path.exists(self.data_path):
            print("数据文件不存在，创建示例数据...")
            create_sample_data(self.data_path)
        
        # 加载数据集
        self.dataset = AlibabaDataset(
            data_path=self.data_path,
            behavior_types=behavior_types,
            test_size=test_size,
            random_state=42
        )
        
        # 更新模型配置
        self.model_config.update({
            'num_users': self.dataset.num_users,
            'num_items': self.dataset.num_items
        })
        
        print(f"数据加载完成:")
        print(f"- 用户数量: {self.dataset.num_users}")
        print(f"- 商品数量: {self.dataset.num_items}")
        print(f"- 交互数量: {self.dataset.num_interactions}")
        
        return self.dataset
    
    def build_model(self):
        """构建模型"""
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        print("正在构建模型...")
        self.model = LightGCN(**self.model_config)
        print(f"模型构建完成: {self.model_config}")
        
        return self.model
    
    def train_model(self, training_config: Dict = None, save_path: str = './models/lightgcn_alibaba.pth'):
        """
        训练模型
        
        Args:
            training_config: 训练配置
            save_path: 模型保存路径
        """
        if self.model is None:
            self.build_model()
        
        # 默认训练配置
        default_config = {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 1024,
            'num_epochs': 100,
            'eval_every': 10,
            'early_stopping_patience': 15
        }
        
        if training_config:
            default_config.update(training_config)
        
        print(f"开始训练模型，配置: {default_config}")
        
        # 创建训练器
        self.trainer = LightGCNTrainer(
            model=self.model,
            dataset=self.dataset,
            learning_rate=default_config['learning_rate'],
            weight_decay=default_config['weight_decay'],
            batch_size=default_config['batch_size'],
            device=self.device
        )
        
        # 开始训练
        history = self.trainer.train(
            num_epochs=default_config['num_epochs'],
            eval_every=default_config['eval_every'],
            early_stopping_patience=default_config['early_stopping_patience'],
            save_path=save_path
        )
        
        return history
    
    def load_trained_model(self, model_path: str):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型配置
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        # 创建模型
        self.model = LightGCN(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        print(f"模型已从 {model_path} 加载")
        
        return self.model
    
    def generate_recommendations(self, user_id: int, k: int = 10, 
                               return_scores: bool = False) -> Tuple[List[int], List[float]]:
        """
        为用户生成推荐
        
        Args:
            user_id: 用户ID
            k: 推荐商品数量
            return_scores: 是否返回得分
            
        Returns:
            推荐商品列表和得分列表
        """
        if self.model is None:
            raise ValueError("模型未加载，请先训练或加载模型")
        
        if self.dataset is None:
            raise ValueError("数据集未加载")
        
        # 获取用户历史交互商品（需要排除）
        user_items = self.dataset.get_user_item_interactions().get(user_id, set())
        
        # 生成推荐
        edge_index = self.dataset.get_edge_index().to(self.device)
        recommended_items, scores = self.model.recommend(
            user_id=user_id,
            edge_index=edge_index,
            k=k,
            exclude_items=user_items
        )
        
        # 转换为列表
        item_list = recommended_items.cpu().numpy().tolist()
        score_list = scores.cpu().numpy().tolist()
        
        if return_scores:
            return item_list, score_list
        else:
            return item_list
    
    def batch_recommend(self, user_ids: List[int], k: int = 10) -> Dict[int, List[int]]:
        """
        批量生成推荐
        
        Args:
            user_ids: 用户ID列表
            k: 推荐商品数量
            
        Returns:
            用户推荐字典
        """
        recommendations = {}
        
        for user_id in user_ids:
            try:
                items = self.generate_recommendations(user_id, k)
                recommendations[user_id] = items
            except Exception as e:
                print(f"为用户 {user_id} 生成推荐时出错: {e}")
                recommendations[user_id] = []
        
        return recommendations
    
    def evaluate_system(self, k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        评估推荐系统性能
        
        Args:
            k_list: 评估的K值列表
            
        Returns:
            评估指标字典
        """
        if self.model is None or self.dataset is None:
            raise ValueError("模型或数据集未加载")
        
        print("正在评估系统性能...")
        
        edge_index = self.dataset.get_edge_index().to(self.device)
        test_data = self.dataset.get_test_data()
        user_item_dict = self.dataset.get_user_item_interactions()
        
        metrics = evaluate_model(
            self.model,
            test_data,
            edge_index,
            user_item_dict,
            k_list,
            self.device
        )
        
        print("评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def analyze_recommendations(self, user_id: int, k: int = 20):
        """
        分析推荐结果
        
        Args:
            user_id: 用户ID
            k: 分析的推荐数量
        """
        if self.dataset is None:
            raise ValueError("数据集未加载")
        
        print(f"\n=== 用户 {user_id} 的推荐分析 ===")
        
        # 获取用户历史交互
        user_items = self.dataset.get_user_item_interactions().get(user_id, set())
        print(f"历史交互商品数量: {len(user_items)}")
        
        if len(user_items) > 0:
            print(f"历史交互商品示例: {list(user_items)[:5]}")
        
        # 生成推荐
        recommended_items, scores = self.generate_recommendations(
            user_id, k, return_scores=True
        )
        
        print(f"\nTop-{k} 推荐商品:")
        for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
            print(f"  {i+1}. 商品ID: {item_id}, 得分: {score:.4f}")
        
        # 计算推荐多样性（不同商品类别的数量）
        if hasattr(self.dataset, 'item_data'):
            recommended_categories = set()
            for item_id in recommended_items:
                # 这里假设可以通过某种方式获取商品类别
                # 实际实现中需要根据数据格式调整
                pass
        
        return recommended_items, scores
    
    def save_recommendations(self, output_path: str, user_ids: List[int] = None, k: int = 10):
        """
        保存推荐结果到文件
        
        Args:
            output_path: 输出文件路径
            user_ids: 用户ID列表，如果为None则为所有用户生成推荐
            k: 推荐商品数量
        """
        if user_ids is None:
            user_ids = list(range(self.dataset.num_users))
        
        print(f"正在为 {len(user_ids)} 个用户生成推荐...")
        
        recommendations = self.batch_recommend(user_ids, k)
        
        # 转换为DataFrame格式
        results = []
        for user_id, items in recommendations.items():
            for rank, item_id in enumerate(items):
                results.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rank': rank + 1
                })
        
        df = pd.DataFrame(results)
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"推荐结果已保存到: {output_path}")
        print(f"总推荐条数: {len(results)}")
        
        return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='阿里移动推荐系统')
    parser.add_argument('--data_path', type=str, default='./alibaba_data', 
                       help='数据文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'recommend', 'evaluate'], 
                       default='train', help='运行模式')
    parser.add_argument('--model_path', type=str, default='./models/lightgcn_alibaba.pth',
                       help='模型文件路径')
    parser.add_argument('--output_path', type=str, default='./results/recommendations.csv',
                       help='推荐结果输出路径')
    parser.add_argument('--embedding_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GCN层数')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--k', type=int, default=10, help='推荐商品数量')
    parser.add_argument('--user_id', type=int, default=0, help='指定用户ID（用于单用户推荐）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型配置
    model_config = {
        'embedding_dim': args.embedding_dim,
        'num_layers': args.num_layers,
        'dropout': 0.1
    }
    
    # 创建推荐系统
    system = AlibabaRecommendationSystem(
        data_path=args.data_path,
        model_config=model_config,
        device=device
    )
    
    # 加载数据
    system.load_data(behavior_types=[4], test_size=0.2)
    
    if args.mode == 'train':
        print("=== 训练模式 ===")
        
        # 训练配置
        training_config = {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'eval_every': 10,
            'early_stopping_patience': 15
        }
        
        # 开始训练
        history = system.train_model(training_config, args.model_path)
        
        # 最终评估
        metrics = system.evaluate_system([5, 10, 20])
        
        # 保存结果
        os.makedirs('./results', exist_ok=True)
        with open('./results/final_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print("训练完成！")
    
    elif args.mode == 'recommend':
        print("=== 推荐模式 ===")
        
        # 加载训练好的模型
        system.load_trained_model(args.model_path)
        
        # 生成推荐（为所有用户或指定用户）
        if args.user_id >= 0:
            # 为指定用户生成推荐
            system.analyze_recommendations(args.user_id, args.k)
        else:
            # 为所有用户生成推荐
            user_ids = list(range(min(100, system.dataset.num_users)))  # 限制数量以节省时间
            system.save_recommendations(args.output_path, user_ids, args.k)
    
    elif args.mode == 'evaluate':
        print("=== 评估模式 ===")
        
        # 加载训练好的模型
        system.load_trained_model(args.model_path)
        
        # 评估系统性能
        metrics = system.evaluate_system([5, 10, 20])
        
        # 保存评估结果
        os.makedirs('./results', exist_ok=True)
        with open('./results/evaluation_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print("评估完成！")


if __name__ == "__main__":
    main()