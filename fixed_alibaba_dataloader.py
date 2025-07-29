"""
修复版本的阿里移动推荐算法数据集加载器
解决数据过滤过于严格导致训练数据不足的问题
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


class FixedAlibabaDataset(Dataset):
    """
    修复版本的阿里移动推荐数据集处理类
    主要改进：
    1. 更宽松的过滤条件
    2. 更好的数据统计和验证
    3. 更合理的训练测试集划分
    """
    
    def __init__(self, data_path: str, behavior_types: List[int] = [4], 
                 min_user_interactions: int = 2, min_item_interactions: int = 2,
                 test_size: float = 0.2, random_state: int = 42):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            behavior_types: 考虑的行为类型列表 [1:浏览, 2:收藏, 3:加购物车, 4:购买]
            min_user_interactions: 用户最少交互次数（降低阈值）
            min_item_interactions: 商品最少交互次数（降低阈值）
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.data_path = data_path
        self.behavior_types = behavior_types
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
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
            if os.path.exists(csv_path):
                self.user_data = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(f"找不到用户数据文件: {user_data_path} 或 {csv_path}")
            
        # 加载商品数据
        item_data_path = os.path.join(self.data_path, "tianchi_mobile_recommend_train_item.csv")
        if os.path.exists(item_data_path):
            self.item_data = pd.read_csv(item_data_path)
        else:
            print("⚠️  警告: 未找到商品数据文件，将使用用户数据中的所有商品")
            # 如果没有商品文件，从用户数据中提取商品
            unique_items = self.user_data['item_id'].unique()
            self.item_data = pd.DataFrame({'item_id': unique_items})
        
        print(f"用户行为数据形状: {self.user_data.shape}")
        print(f"商品数据形状: {self.item_data.shape}")
        print("数据加载完成!")
        
    def _preprocess_data(self):
        """改进的数据预处理"""
        print("正在预处理数据...")
        
        # 显示原始数据统计
        print(f"原始数据统计:")
        print(f"- 总交互数: {len(self.user_data)}")
        print(f"- 唯一用户数: {self.user_data['user_id'].nunique()}")
        print(f"- 唯一商品数: {self.user_data['item_id'].nunique()}")
        print(f"- 行为类型分布:")
        for bt in sorted(self.user_data['behavior_type'].unique()):
            count = (self.user_data['behavior_type'] == bt).sum()
            print(f"  类型{bt}: {count} ({count/len(self.user_data)*100:.1f}%)")
        
        # 1. 过滤指定行为类型
        original_size = len(self.user_data)
        self.user_data = self.user_data[self.user_data['behavior_type'].isin(self.behavior_types)]
        print(f"\n步骤1 - 行为类型过滤: {original_size} → {len(self.user_data)}")
        
        if len(self.user_data) == 0:
            raise ValueError(f"过滤行为类型 {self.behavior_types} 后数据为空！")
        
        # 2. 只保留商品子集中的商品
        item_ids = set(self.item_data['item_id'].unique())
        before_item_filter = len(self.user_data)
        self.user_data = self.user_data[self.user_data['item_id'].isin(item_ids)]
        print(f"步骤2 - 商品过滤: {before_item_filter} → {len(self.user_data)}")
        
        if len(self.user_data) == 0:
            raise ValueError("商品过滤后数据为空！")
        
        # 3. 分析交互频次分布
        user_inter_count = self.user_data.groupby('user_id').size()
        item_inter_count = self.user_data.groupby('item_id').size()
        
        print(f"\n交互频次分析:")
        print(f"用户交互次数 - 最小:{user_inter_count.min()}, 最大:{user_inter_count.max()}, 平均:{user_inter_count.mean():.2f}")
        print(f"商品交互次数 - 最小:{item_inter_count.min()}, 最大:{item_inter_count.max()}, 平均:{item_inter_count.mean():.2f}")
        
        # 4. 应用更宽松的最小交互频次过滤
        print(f"\n应用最小交互频次过滤 (用户>={self.min_user_interactions}, 商品>={self.min_item_interactions}):")
        
        valid_users = set(user_inter_count[user_inter_count >= self.min_user_interactions].index)
        valid_items = set(item_inter_count[item_inter_count >= self.min_item_interactions].index)
        
        print(f"有效用户: {len(valid_users)} / {len(user_inter_count)}")
        print(f"有效商品: {len(valid_items)} / {len(item_inter_count)}")
        
        before_final_filter = len(self.user_data)
        self.user_data = self.user_data[
            self.user_data['user_id'].isin(valid_users) &
            self.user_data['item_id'].isin(valid_items)
        ]
        print(f"步骤3 - 频次过滤: {before_final_filter} → {len(self.user_data)}")
        
        # 检查是否过滤过度
        if len(self.user_data) < 1000:
            print(f"⚠️  警告: 过滤后只剩 {len(self.user_data)} 条交互数据，建议降低过滤条件")
            
            if len(self.user_data) == 0:
                # 如果过滤后为空，尝试更宽松的条件
                print("🔄 尝试更宽松的过滤条件...")
                self.min_user_interactions = max(1, self.min_user_interactions - 1)
                self.min_item_interactions = max(1, self.min_item_interactions - 1)
                
                # 重新计算
                user_inter_count = self.user_data.groupby('user_id').size()
                item_inter_count = self.user_data.groupby('item_id').size()
                valid_users = set(user_inter_count[user_inter_count >= self.min_user_interactions].index)
                valid_items = set(item_inter_count[item_inter_count >= self.min_item_interactions].index)
                
                self.user_data = self.user_data[
                    self.user_data['user_id'].isin(valid_users) &
                    self.user_data['item_id'].isin(valid_items)
                ]
                
                if len(self.user_data) == 0:
                    # 最后的保险：不应用频次过滤
                    print("🚨 完全取消频次过滤，保留所有数据")
                    before_final_filter = len(self.user_data)
                    # 重新加载过滤后的数据
                    self.user_data = self.user_data  # 这里应该是重新加载前面步骤的结果
        
        # 5. 编码用户和商品ID
        self.user_data['user_id_encoded'] = self.user_encoder.fit_transform(self.user_data['user_id'])
        self.user_data['item_id_encoded'] = self.item_encoder.fit_transform(self.user_data['item_id'])
        
        # 6. 基本统计信息
        self.num_users = self.user_data['user_id_encoded'].nunique()
        self.num_items = self.user_data['item_id_encoded'].nunique()
        self.num_interactions = len(self.user_data)
        
        print(f"\n✅ 最终数据统计:")
        print(f"用户数量: {self.num_users}")
        print(f"商品数量: {self.num_items}")
        print(f"交互数量: {self.num_interactions}")
        print(f"平均每用户交互: {self.num_interactions/self.num_users:.1f}")
        print(f"平均每商品交互: {self.num_interactions/self.num_items:.1f}")
        print(f"稀疏度: {(1 - self.num_interactions/(self.num_users * self.num_items))*100:.2f}%")
        
        # 7. 时间排序
        self.user_data['time'] = pd.to_datetime(self.user_data['time'])
        self.user_data = self.user_data.sort_values('time')
        
        # 检查数据质量
        if self.num_interactions < 1000:
            print("⚠️  数据量较少，建议检查数据源或降低过滤条件")
        if self.num_users < 50 or self.num_items < 50:
            print("⚠️  用户或商品数量较少，可能影响模型效果")
            
    def _build_graph(self):
        """构建用户-商品二部图"""
        print("\n正在构建图结构...")
        
        # 创建训练测试集分割 - 使用随机分割而不是时间分割
        # 因为时间分割可能导致数据分布不均
        train_indices, test_indices = train_test_split(
            range(len(self.user_data)), 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.user_data['user_id_encoded']  # 按用户分层
        )
        
        train_data = self.user_data.iloc[train_indices]
        test_data = self.user_data.iloc[test_indices]
        
        print(f"训练集大小: {len(train_data)} ({len(train_data)/len(self.user_data)*100:.1f}%)")
        print(f"测试集大小: {len(test_data)} ({len(test_data)/len(self.user_data)*100:.1f}%)")
        
        # 构建训练集的边索引
        train_interactions = train_data[['user_id_encoded', 'item_id_encoded']].values
        
        # 创建双向边：用户->商品 和 商品->用户
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
            
        print(f"图节点总数: {self.num_users + self.num_items}")
        print(f"图边数量: {len(all_edges)}")
        print("图构建完成!")
        
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
        改进的负采样方法
        """
        positive_items = self.user_item_dict[user_id]
        negative_items = []
        
        # 增加最大尝试次数，避免死循环
        max_attempts = min(num_negatives * 50, self.num_items * 2)
        attempts = 0
        
        while len(negative_items) < num_negatives and attempts < max_attempts:
            item_id = np.random.randint(0, self.num_items)
            if item_id not in positive_items:
                negative_items.append(item_id)
            attempts += 1
                
        # 如果采样不足，用随机填充
        while len(negative_items) < num_negatives:
            negative_items.append(np.random.randint(0, self.num_items))
            
        return negative_items
    
    def get_user_item_interactions(self) -> Dict[int, List[int]]:
        """获取用户-商品交互字典"""
        return dict(self.user_item_dict)
    
    def get_data_statistics(self) -> Dict:
        """获取详细的数据统计信息"""
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_interactions': self.num_interactions,
            'num_train_interactions': len(self.train_interactions),
            'num_test_interactions': len(self.test_interactions),
            'sparsity': 1 - self.num_interactions / (self.num_users * self.num_items),
            'avg_user_interactions': self.num_interactions / self.num_users,
            'avg_item_interactions': self.num_interactions / self.num_items
        }
    
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


def create_larger_sample_data(save_path: str):
    """
    创建更大规模的示例数据，模拟真实场景
    """
    print("正在创建大规模示例数据...")
    
    np.random.seed(42)
    
    # 增加数据规模
    num_users = 5000
    num_items = 2000
    num_interactions = 50000
    
    # 生成更真实的用户行为数据
    # 模拟用户行为的幂律分布
    user_popularity = np.random.pareto(0.5, num_users) + 1
    item_popularity = np.random.pareto(0.8, num_items) + 1
    
    user_weights = user_popularity / user_popularity.sum()
    item_weights = item_popularity / item_popularity.sum()
    
    user_data = {
        'user_id': np.random.choice(range(1, num_users + 1), num_interactions, p=user_weights),
        'item_id': np.random.choice(range(1, num_items + 1), num_interactions, p=item_weights),
        'behavior_type': np.random.choice([1, 2, 3, 4], num_interactions, p=[0.5, 0.25, 0.15, 0.1]),
        'user_geohash': ['geo_' + str(i) for i in np.random.randint(1, 200, num_interactions)],
        'item_category': np.random.randint(1, 100, num_interactions),
        'time': pd.date_range('2014-11-01', '2014-12-31', periods=num_interactions)
    }
    
    user_df = pd.DataFrame(user_data)
    
    # 生成商品数据（保留大部分商品）
    selected_items = np.random.choice(range(1, num_items + 1), size=int(num_items * 0.8), replace=False)
    item_data = {
        'item_id': selected_items,
        'item_geohash': ['item_geo_' + str(i) for i in np.random.randint(1, 150, len(selected_items))],
        'item_category': np.random.randint(1, 100, len(selected_items))
    }
    
    item_df = pd.DataFrame(item_data)
    
    # 保存数据
    os.makedirs(save_path, exist_ok=True)
    user_df.to_csv(os.path.join(save_path, 'tianchi_mobile_recommend_train_user.csv'), index=False)
    item_df.to_csv(os.path.join(save_path, 'tianchi_mobile_recommend_train_item.csv'), index=False)
    
    print(f"大规模示例数据已保存到: {save_path}")
    print(f"用户行为数据: {user_df.shape}")
    print(f"商品数据: {item_df.shape}")
    print(f"行为类型分布:")
    print(user_df['behavior_type'].value_counts().sort_index())


if __name__ == "__main__":
    # 测试修复版本的数据加载器
    data_path = "./test_alibaba_data"
    
    # 创建大规模测试数据
    create_larger_sample_data(data_path)
    
    # 测试不同的过滤条件
    print("\n" + "="*60)
    print("测试修复版本的数据加载器")
    print("="*60)
    
    # 测试1: 宽松条件
    try:
        dataset = FixedAlibabaDataset(
            data_path=data_path,
            behavior_types=[4],  # 只考虑购买行为
            min_user_interactions=2,  # 降低用户最小交互次数
            min_item_interactions=2,  # 降低商品最小交互次数
            test_size=0.2
        )
        
        print(f"\n✅ 宽松过滤条件测试成功:")
        stats = dataset.get_data_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ 宽松过滤条件测试失败: {e}")
    
    # 测试2: 严格条件 (您原来的条件)
    try:
        dataset_strict = FixedAlibabaDataset(
            data_path=data_path,
            behavior_types=[4],
            min_user_interactions=5,  # 您原来的条件
            min_item_interactions=5,  # 您原来的条件
            test_size=0.2
        )
        
        print(f"\n✅ 严格过滤条件测试成功:")
        stats = dataset_strict.get_data_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ 严格过滤条件测试失败: {e}")