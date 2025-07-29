"""
阿里移动推荐算法数据集加载器
用于处理阿里移动推荐挑战赛的用户行为数据和商品数据
支持LightGCN等图神经网络模型的数据格式
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import zipfile
from typing import Dict, Tuple, List
from collections import defaultdict


class AlibabaDataset(Dataset):
    """
    阿里移动推荐数据集处理类
    
    处理tianchi_mobile_recommend_train_user.zip和tianchi_mobile_recommend_train_item.csv文件
    生成用户-商品交互图的边索引，支持图神经网络训练
    """
    
    def __init__(self, data_path: str, behavior_types: List[int] = [4], test_size: float = 0.2, random_state: int = 42):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            behavior_types: 考虑的行为类型列表 [1:浏览, 2:收藏, 3:加购物车, 4:购买]
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.data_path = data_path
        self.behavior_types = behavior_types
        self.test_size = test_size
        self.random_state = random_state
        
        # 初始化编码器
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # 加载和预处理数据
        self._load_data()
        self._preprocess_data()
        self._build_graph()
        
    def _load_data(self):
        """加载原始数据文件"""
        print("正在加载数据文件...")
        
        # 加载用户行为数据
        user_data_path = os.path.join(self.data_path, "tianchi_mobile_recommend_train_user.zip")
        if os.path.exists(user_data_path):
            # 如果是zip文件，解压读取
            with zipfile.ZipFile(user_data_path, 'r') as zip_file:
                csv_file = zip_file.namelist()[0]
                self.user_data = pd.read_csv(zip_file.open(csv_file))
        else:
            # 如果已经解压，直接读取CSV
            csv_path = os.path.join(self.data_path, "tianchi_mobile_recommend_train_user.csv")
            self.user_data = pd.read_csv(csv_path)
            
        # 加载商品数据
        item_data_path = os.path.join(self.data_path, "tianchi_mobile_recommend_train_item.csv")
        self.item_data = pd.read_csv(item_data_path)
        
        print(f"用户行为数据形状: {self.user_data.shape}")
        print(f"商品数据形状: {self.item_data.shape}")
        print("数据加载完成!")
        
    def _preprocess_data(self):
        """预处理数据"""
        print("正在预处理数据...")
        
        # 过滤指定的行为类型
        self.user_data = self.user_data[self.user_data['behavior_type'].isin(self.behavior_types)]
        
        # 只保留商品子集中的商品
        item_ids = set(self.item_data['item_id'].unique())
        self.user_data = self.user_data[self.user_data['item_id'].isin(item_ids)]
        
        # 编码用户ID和商品ID
        all_users = self.user_data['user_id'].unique()
        all_items = self.user_data['item_id'].unique()
        
        self.user_data['user_id_encoded'] = self.user_encoder.fit_transform(self.user_data['user_id'])
        self.user_data['item_id_encoded'] = self.item_encoder.fit_transform(self.user_data['item_id'])
        
        # 存储基本统计信息
        self.num_users = len(all_users)
        self.num_items = len(all_items)
        self.num_interactions = len(self.user_data)
        
        print(f"用户数量: {self.num_users}")
        print(f"商品数量: {self.num_items}")
        print(f"交互数量: {self.num_interactions}")
        
        # 创建用户-商品交互矩阵
        self.interactions = self.user_data[['user_id_encoded', 'item_id_encoded']].values
        
        # 按时间排序，用于划分训练测试集
        self.user_data['time'] = pd.to_datetime(self.user_data['time'])
        self.user_data = self.user_data.sort_values('time')
        
    def _build_graph(self):
        """构建用户-商品二部图"""
        print("正在构建图结构...")
        
        # 创建训练测试集分割
        # 使用时间划分：最后一天的数据作为测试集
        test_date = self.user_data['time'].max().date()
        train_data = self.user_data[self.user_data['time'].dt.date < test_date]
        test_data = self.user_data[self.user_data['time'].dt.date == test_date]
        
        # 如果最后一天数据太少，使用随机划分
        if len(test_data) < 0.1 * len(self.user_data):
            train_indices, test_indices = train_test_split(
                range(len(self.user_data)), 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            train_data = self.user_data.iloc[train_indices]
            test_data = self.user_data.iloc[test_indices]
        
        # 构建训练集的边索引 (edge_index)
        train_interactions = train_data[['user_id_encoded', 'item_id_encoded']].values
        
        # 创建双向边：用户->商品 和 商品->用户
        # 商品节点的索引需要偏移，避免与用户节点索引重复
        user_to_item = train_interactions
        item_to_user = np.column_stack([
            train_interactions[:, 1] + self.num_users,  # 商品节点索引偏移
            train_interactions[:, 0]  # 用户节点索引
        ])
        
        # 合并所有边
        all_edges = np.vstack([user_to_item, item_to_user])
        
        # 转换为PyTorch Geometric格式的边索引
        self.edge_index = torch.from_numpy(all_edges.T).long()
        
        # 存储训练和测试数据
        self.train_interactions = train_interactions
        self.test_interactions = test_data[['user_id_encoded', 'item_id_encoded']].values
        
        # 创建用户的历史交互记录，用于负采样
        self.user_item_dict = defaultdict(set)
        for user_id, item_id in train_interactions:
            self.user_item_dict[user_id].add(item_id)
            
        print(f"训练交互数量: {len(train_interactions)}")
        print(f"测试交互数量: {len(self.test_interactions)}")
        print(f"图节点总数: {self.num_users + self.num_items}")
        print(f"图边数量: {len(all_edges)}")
        
    def get_edge_index(self):
        """获取图的边索引"""
        return self.edge_index
    
    def get_num_nodes(self):
        """获取图的节点总数"""
        return self.num_users + self.num_items
    
    def get_train_data(self):
        """获取训练数据"""
        return self.train_interactions
    
    def get_test_data(self):
        """获取测试数据"""
        return self.test_interactions
    
    def negative_sampling(self, user_id: int, num_negatives: int = 1) -> List[int]:
        """
        为指定用户进行负采样
        
        Args:
            user_id: 用户ID
            num_negatives: 负样本数量
            
        Returns:
            负样本商品ID列表
        """
        positive_items = self.user_item_dict[user_id]
        negative_items = []
        
        while len(negative_items) < num_negatives:
            # 随机选择商品
            item_id = np.random.randint(0, self.num_items)
            # 确保不是正样本
            if item_id not in positive_items:
                negative_items.append(item_id)
                
        return negative_items
    
    def get_user_item_interactions(self) -> Dict[int, List[int]]:
        """获取用户-商品交互字典"""
        return dict(self.user_item_dict)
    
    def __len__(self):
        """返回训练样本数量"""
        return len(self.train_interactions)
    
    def __getitem__(self, idx):
        """获取单个训练样本"""
        user_id, item_id = self.train_interactions[idx]
        
        # 正样本
        positive_sample = {
            'user_id': user_id,
            'item_id': item_id,
            'label': 1.0
        }
        
        # 负样本
        negative_item = self.negative_sampling(user_id, 1)[0]
        negative_sample = {
            'user_id': user_id,
            'item_id': negative_item,
            'label': 0.0
        }
        
        return positive_sample, negative_sample


def create_sample_data(save_path: str):
    """
    创建示例数据文件，用于测试
    如果没有真实的阿里数据集，可以使用这个函数生成模拟数据
    
    Args:
        save_path: 保存路径
    """
    print("正在创建示例数据...")
    
    # 创建示例用户行为数据
    np.random.seed(42)
    
    num_users = 1000
    num_items = 500
    num_interactions = 10000
    
    # 生成用户行为数据
    user_data = {
        'user_id': np.random.randint(1, num_users + 1, num_interactions),
        'item_id': np.random.randint(1, num_items + 1, num_interactions),
        'behavior_type': np.random.choice([1, 2, 3, 4], num_interactions, p=[0.6, 0.15, 0.15, 0.1]),
        'user_geohash': ['geo_' + str(i) for i in np.random.randint(1, 100, num_interactions)],
        'item_category': np.random.randint(1, 50, num_interactions),
        'time': pd.date_range('2014-11-18', '2014-12-19', periods=num_interactions)
    }
    
    user_df = pd.DataFrame(user_data)
    
    # 生成商品数据（商品子集）
    selected_items = np.random.choice(range(1, num_items + 1), size=num_items // 2, replace=False)
    item_data = {
        'item_id': selected_items,
        'item_geohash': ['item_geo_' + str(i) for i in np.random.randint(1, 100, len(selected_items))],
        'item_category': np.random.randint(1, 50, len(selected_items))
    }
    
    item_df = pd.DataFrame(item_data)
    
    # 保存数据
    os.makedirs(save_path, exist_ok=True)
    user_df.to_csv(os.path.join(save_path, 'tianchi_mobile_recommend_train_user.csv'), index=False)
    item_df.to_csv(os.path.join(save_path, 'tianchi_mobile_recommend_train_item.csv'), index=False)
    
    print(f"示例数据已保存到: {save_path}")
    print(f"用户行为数据: {user_df.shape}")
    print(f"商品数据: {item_df.shape}")


if __name__ == "__main__":
    # 测试数据加载器
    data_path = "./alibaba_data"
    
    # 如果没有真实数据，创建示例数据
    if not os.path.exists(data_path):
        create_sample_data(data_path)
    
    # 加载数据集
    dataset = AlibabaDataset(
        data_path=data_path,
        behavior_types=[4],  # 只考虑购买行为
        test_size=0.2
    )
    
    print("\n数据集信息:")
    print(f"用户数量: {dataset.num_users}")
    print(f"商品数量: {dataset.num_items}")
    print(f"总节点数: {dataset.get_num_nodes()}")
    print(f"边索引形状: {dataset.get_edge_index().shape}")
    print(f"训练样本数: {len(dataset)}")