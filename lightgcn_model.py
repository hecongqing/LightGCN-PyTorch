"""
LightGCN模型实现
基于PyTorch Geometric实现的轻量级图卷积网络用于推荐系统

LightGCN论文: https://arxiv.org/abs/2002.02126
"LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"

主要特点:
1. 简化的图卷积操作，只保留邻域聚合
2. 去除特征变换和非线性激活
3. 多层嵌入的加权平均
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import numpy as np
from typing import Optional, Tuple


class LightGCNConv(MessagePassing):
    """
    LightGCN卷积层
    
    简化的图卷积操作，只进行邻域聚合，不包含特征变换和激活函数
    消息传递公式: h_i^(k+1) = Σ_{j∈N(i)} 1/√(|N(i)||N(j)|) * h_j^(k)
    """
    
    def __init__(self, **kwargs):
        """初始化LightGCN卷积层"""
        super().__init__(aggr='add', **kwargs)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征矩阵 [num_nodes, feature_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
            
        Returns:
            更新后的节点特征 [num_nodes, feature_dim]
        """
        # 计算归一化系数
        if edge_weight is None:
            # 计算节点度数
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            # 对称归一化: D^(-1/2) A D^(-1/2)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = edge_weight
        
        # 执行消息传递
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        消息函数：计算从邻居节点j到当前节点的消息
        
        Args:
            x_j: 邻居节点特征 [num_edges, feature_dim]
            norm: 归一化系数 [num_edges]
            
        Returns:
            消息 [num_edges, feature_dim]
        """
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """
    LightGCN推荐模型
    
    使用轻量级图卷积网络学习用户和商品的嵌入表示
    适用于协同过滤推荐任务
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 num_layers: int = 3, dropout: float = 0.1):
        """
        初始化LightGCN模型
        
        Args:
            num_users: 用户数量
            num_items: 商品数量
            embedding_dim: 嵌入维度
            num_layers: 图卷积层数
            dropout: Dropout比例
        """
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 用户和商品的嵌入层
        # 总节点数 = 用户数 + 商品数
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        
        # LightGCN卷积层
        self.convs = nn.ModuleList([
            LightGCNConv() for _ in range(num_layers)
        ])
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化嵌入参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.embedding.weight)
        
    def forward(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            edge_index: 图的边索引 [2, num_edges]
            
        Returns:
            用户嵌入和商品嵌入的元组
        """
        # 获取所有节点的初始嵌入
        x = self.embedding.weight
        
        # 存储每一层的嵌入
        embeddings = [x]
        
        # 多层图卷积
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.dropout_layer(x)
            embeddings.append(x)
        
        # 对所有层的嵌入进行加权平均
        # LightGCN的关键：简单平均所有层的嵌入
        final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        
        # 分离用户和商品嵌入
        user_embeddings = final_embedding[:self.num_users]
        item_embeddings = final_embedding[self.num_users:]
        
        return user_embeddings, item_embeddings
    
    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        预测用户对商品的偏好得分
        
        Args:
            user_ids: 用户ID列表 [batch_size]
            item_ids: 商品ID列表 [batch_size]
            edge_index: 图的边索引
            
        Returns:
            预测得分 [batch_size]
        """
        # 获取用户和商品嵌入
        user_embeddings, item_embeddings = self.forward(edge_index)
        
        # 选择对应的用户和商品嵌入
        user_emb = user_embeddings[user_ids]  # [batch_size, embedding_dim]
        item_emb = item_embeddings[item_ids]  # [batch_size, embedding_dim]
        
        # 计算内积得分
        scores = torch.sum(user_emb * item_emb, dim=1)  # [batch_size]
        
        return scores
    
    def get_user_embedding(self, user_id: int, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取特定用户的嵌入
        
        Args:
            user_id: 用户ID
            edge_index: 图的边索引
            
        Returns:
            用户嵌入向量
        """
        user_embeddings, _ = self.forward(edge_index)
        return user_embeddings[user_id]
    
    def get_item_embedding(self, item_id: int, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取特定商品的嵌入
        
        Args:
            item_id: 商品ID
            edge_index: 图的边索引
            
        Returns:
            商品嵌入向量
        """
        _, item_embeddings = self.forward(edge_index)
        return item_embeddings[item_id]
    
    def recommend(self, user_id: int, edge_index: torch.Tensor, 
                  k: int = 10, exclude_items: Optional[set] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为用户推荐Top-K商品
        
        Args:
            user_id: 用户ID
            edge_index: 图的边索引
            k: 推荐商品数量
            exclude_items: 需要排除的商品ID集合（如用户已交互的商品）
            
        Returns:
            推荐商品ID和对应得分的元组
        """
        with torch.no_grad():
            # 获取用户和所有商品的嵌入
            user_embeddings, item_embeddings = self.forward(edge_index)
            
            # 计算用户与所有商品的得分
            user_emb = user_embeddings[user_id].unsqueeze(0)  # [1, embedding_dim]
            scores = torch.matmul(user_emb, item_embeddings.t()).squeeze()  # [num_items]
            
            # 排除已交互的商品
            if exclude_items:
                for item_id in exclude_items:
                    scores[item_id] = float('-inf')
            
            # 获取Top-K商品
            top_scores, top_items = torch.topk(scores, k)
            
            return top_items, top_scores


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) 损失函数
    
    专门为推荐系统设计的排序损失函数
    优化目标：正样本得分 > 负样本得分
    """
    
    def __init__(self, lambda_reg: float = 1e-4):
        """
        初始化BPR损失
        
        Args:
            lambda_reg: L2正则化系数
        """
        super(BPRLoss, self).__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, 
                model: LightGCN) -> torch.Tensor:
        """
        计算BPR损失
        
        Args:
            pos_scores: 正样本得分 [batch_size]
            neg_scores: 负样本得分 [batch_size]
            model: LightGCN模型，用于计算正则化项
            
        Returns:
            总损失值
        """
        # BPR损失：-log(sigmoid(pos_score - neg_score))
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        # L2正则化
        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.norm(param, p=2) ** 2
        
        reg_loss = self.lambda_reg * reg_loss
        
        total_loss = bpr_loss + reg_loss
        
        return total_loss


def evaluate_model(model: LightGCN, test_data: np.ndarray, edge_index: torch.Tensor,
                   user_item_dict: dict, k_list: list = [10, 20], device: str = 'cpu') -> dict:
    """
    评估模型性能
    
    Args:
        model: 训练好的LightGCN模型
        test_data: 测试数据 [num_test, 2] (user_id, item_id)
        edge_index: 图的边索引
        user_item_dict: 用户历史交互字典
        k_list: 评估的K值列表
        device: 设备
        
    Returns:
        评估指标字典
    """
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        # 按用户分组测试数据
        user_test_dict = {}
        for user_id, item_id in test_data:
            if user_id not in user_test_dict:
                user_test_dict[user_id] = []
            user_test_dict[user_id].append(item_id)
        
        # 计算评估指标
        for k in k_list:
            precision_list = []
            recall_list = []
            ndcg_list = []
            
            for user_id, true_items in user_test_dict.items():
                # 获取推荐列表
                exclude_items = user_item_dict.get(user_id, set())
                recommended_items, _ = model.recommend(
                    user_id, edge_index, k=k, exclude_items=exclude_items
                )
                
                recommended_items = recommended_items.cpu().numpy()
                true_items = set(true_items)
                recommended_set = set(recommended_items)
                
                # 计算Precision@K
                precision = len(true_items & recommended_set) / k
                precision_list.append(precision)
                
                # 计算Recall@K
                recall = len(true_items & recommended_set) / len(true_items) if len(true_items) > 0 else 0
                recall_list.append(recall)
                
                # 计算NDCG@K
                dcg = 0
                idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
                
                for i, item in enumerate(recommended_items):
                    if item in true_items:
                        dcg += 1 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_list.append(ndcg)
            
            # 计算平均指标
            results[f'Precision@{k}'] = np.mean(precision_list)
            results[f'Recall@{k}'] = np.mean(recall_list)
            results[f'NDCG@{k}'] = np.mean(ndcg_list)
    
    return results


if __name__ == "__main__":
    # 测试模型
    print("测试LightGCN模型...")
    
    # 模型参数
    num_users = 1000
    num_items = 500
    embedding_dim = 64
    num_layers = 3
    
    # 创建模型
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    )
    
    # 创建示例边索引
    num_edges = 5000
    edge_index = torch.randint(0, num_users + num_items, (2, num_edges))
    
    # 测试前向传播
    user_embeddings, item_embeddings = model(edge_index)
    print(f"用户嵌入形状: {user_embeddings.shape}")
    print(f"商品嵌入形状: {item_embeddings.shape}")
    
    # 测试预测
    user_ids = torch.randint(0, num_users, (32,))
    item_ids = torch.randint(0, num_items, (32,))
    scores = model.predict(user_ids, item_ids, edge_index)
    print(f"预测得分形状: {scores.shape}")
    
    # 测试推荐
    top_items, top_scores = model.recommend(0, edge_index, k=10)
    print(f"推荐商品: {top_items}")
    print(f"推荐得分: {top_scores}")
    
    print("模型测试完成!")