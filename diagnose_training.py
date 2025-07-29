"""
诊断阿里巴巴数据集训练loss不动的问题
检查数据预处理、模型设置、训练过程等各个环节
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def create_test_data():
    """创建测试数据来诊断问题"""
    print("=== 创建测试数据 ===")
    
    # 创建模拟的阿里巴巴数据格式
    np.random.seed(42)
    
    num_users_raw = 2000
    num_items_raw = 1000
    num_interactions = 15000
    
    # 生成用户行为数据
    user_data = pd.DataFrame({
        'user_id': np.random.randint(1, num_users_raw + 1, num_interactions),
        'item_id': np.random.randint(1, num_items_raw + 1, num_interactions),
        'behavior_type': np.random.choice([1, 2, 3, 4], num_interactions, p=[0.5, 0.2, 0.2, 0.1]),
        'time': pd.date_range('2014-11-18', '2014-12-19', periods=num_interactions)
    })
    
    # 生成商品数据（只保留部分商品）
    selected_items = np.random.choice(range(1, num_items_raw + 1), size=500, replace=False)
    item_data = pd.DataFrame({
        'item_id': selected_items
    })
    
    return user_data, item_data

def analyze_data_preprocessing(user_data, item_data, behavior_types=[4]):
    """分析数据预处理步骤"""
    print("\n=== 数据预处理分析 ===")
    
    print(f"原始用户行为数据: {user_data.shape}")
    print(f"原始商品数据: {item_data.shape}")
    print(f"原始行为类型分布:")
    print(user_data['behavior_type'].value_counts().sort_index())
    
    # 1. 过滤指定行为类型
    user_data_filtered = user_data[user_data['behavior_type'].isin(behavior_types)]
    print(f"\n过滤后数据: {user_data_filtered.shape}")
    
    # 2. 只保留商品子集中的商品
    item_ids = set(item_data['item_id'].unique())
    user_data_filtered = user_data_filtered[user_data_filtered['item_id'].isin(item_ids)]
    print(f"商品过滤后数据: {user_data_filtered.shape}")
    
    # 3. 检查用户和商品的交互频次分布
    user_inter_count = user_data_filtered.groupby('user_id').size()
    item_inter_count = user_data_filtered.groupby('item_id').size()
    
    print(f"\n用户交互频次统计:")
    print(f"最小值: {user_inter_count.min()}, 最大值: {user_inter_count.max()}, 平均值: {user_inter_count.mean():.2f}")
    print(f"少于5次交互的用户: {(user_inter_count < 5).sum()} / {len(user_inter_count)} ({(user_inter_count < 5).mean()*100:.1f}%)")
    
    print(f"\n商品交互频次统计:")
    print(f"最小值: {item_inter_count.min()}, 最大值: {item_inter_count.max()}, 平均值: {item_inter_count.mean():.2f}")
    print(f"少于5次交互的商品: {(item_inter_count < 5).sum()} / {len(item_inter_count)} ({(item_inter_count < 5).mean()*100:.1f}%)")
    
    # 4. 应用最小交互频次过滤 (您的代码中的步骤)
    N = 5
    valid_users = set(user_inter_count[user_inter_count >= N].index)
    valid_items = set(item_inter_count[item_inter_count >= N].index)
    
    user_data_final = user_data_filtered[
        user_data_filtered['user_id'].isin(valid_users) &
        user_data_filtered['item_id'].isin(valid_items)
    ]
    
    print(f"\n最终数据 (最少{N}次交互): {user_data_final.shape}")
    
    if len(user_data_final) == 0:
        print("❌ 警告: 过滤后数据为空！这会导致训练无法进行。")
        return None
    
    # 5. 编码用户和商品ID
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    user_data_final['user_id_encoded'] = user_encoder.fit_transform(user_data_final['user_id'])
    user_data_final['item_id_encoded'] = item_encoder.fit_transform(user_data_final['item_id'])
    
    num_users = user_data_final['user_id_encoded'].nunique()
    num_items = user_data_final['item_id_encoded'].nunique()
    num_interactions = len(user_data_final)
    
    print(f"\n最终统计:")
    print(f"用户数量: {num_users}")
    print(f"商品数量: {num_items}")
    print(f"交互数量: {num_interactions}")
    print(f"稀疏度: {(1 - num_interactions / (num_users * num_items)) * 100:.2f}%")
    
    # 检查数据质量
    if num_interactions < 1000:
        print("❌ 警告: 交互数量过少，可能导致训练不稳定")
    if num_users < 100 or num_items < 100:
        print("❌ 警告: 用户或商品数量过少，可能导致模型容量不足")
    
    return user_data_final, num_users, num_items

def check_model_and_loss():
    """检查模型和损失函数"""
    print("\n=== 模型和损失检查 ===")
    
    # 创建小规模测试模型
    num_users, num_items = 100, 50
    embedding_dim = 16
    
    # 简化的测试模型
    class SimpleLightGCN(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim):
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
            nn.init.xavier_uniform_(self.embedding.weight)
            
        def forward(self, edge_index):
            x = self.embedding.weight
            return x[:self.num_users], x[self.num_users:]
            
        def predict(self, user_ids, item_ids, edge_index):
            user_emb, item_emb = self.forward(edge_index)
            user_emb = user_emb[user_ids]
            item_emb = item_emb[item_ids]
            return torch.sum(user_emb * item_emb, dim=1)
    
    model = SimpleLightGCN(num_users, num_items, embedding_dim)
    
    # 测试前向传播
    batch_size = 32
    user_ids = torch.randint(0, num_users, (batch_size,))
    pos_item_ids = torch.randint(0, num_items, (batch_size,))
    neg_item_ids = torch.randint(0, num_items, (batch_size,))
    
    # 创建虚拟边索引
    edge_index = torch.tensor([[0, 1, 2], [num_users, num_users+1, num_users+2]], dtype=torch.long)
    
    # 计算预测得分
    pos_scores = model.predict(user_ids, pos_item_ids, edge_index)
    neg_scores = model.predict(user_ids, neg_item_ids, edge_index)
    
    print(f"正样本得分统计: min={pos_scores.min():.4f}, max={pos_scores.max():.4f}, mean={pos_scores.mean():.4f}")
    print(f"负样本得分统计: min={neg_scores.min():.4f}, max={neg_scores.max():.4f}, mean={neg_scores.mean():.4f}")
    print(f"得分差异: {(pos_scores - neg_scores).mean():.4f}")
    
    # 测试BPR损失
    class SimpleBPRLoss(nn.Module):
        def __init__(self, lambda_reg=1e-4):
            super().__init__()
            self.lambda_reg = lambda_reg
            
        def forward(self, pos_scores, neg_scores, model):
            # BPR损失
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            
            # L2正则化
            reg_loss = 0
            for param in model.parameters():
                reg_loss += torch.norm(param, p=2) ** 2
            reg_loss = self.lambda_reg * reg_loss
            
            return bpr_loss + reg_loss, bpr_loss.item(), reg_loss.item()
    
    criterion = SimpleBPRLoss()
    total_loss, bpr_loss, reg_loss = criterion(pos_scores, neg_scores, model)
    
    print(f"\n损失分析:")
    print(f"BPR损失: {bpr_loss:.6f}")
    print(f"正则化损失: {reg_loss:.6f}")
    print(f"总损失: {total_loss.item():.6f}")
    
    # 检查梯度
    total_loss.backward()
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            print(f"{name} 梯度范数: {grad_norm:.6f}")
    
    if all(g < 1e-6 for g in grad_norms):
        print("❌ 警告: 梯度过小，可能导致学习停滞")
    
    return model

def simulate_training_process():
    """模拟训练过程，检查常见问题"""
    print("\n=== 训练过程模拟 ===")
    
    # 使用之前的测试数据
    user_data, item_data = create_test_data()
    processed_data = analyze_data_preprocessing(user_data, item_data)
    
    if processed_data is None:
        print("❌ 数据预处理失败，无法继续训练模拟")
        return
    
    user_data_final, num_users, num_items = processed_data
    
    # 创建训练数据
    interactions = user_data_final[['user_id_encoded', 'item_id_encoded']].values
    print(f"训练交互数: {len(interactions)}")
    
    # 模拟负采样
    def negative_sampling(user_id, user_item_dict, num_items, num_negatives=1):
        positive_items = user_item_dict.get(user_id, set())
        negative_items = []
        max_attempts = num_negatives * 10
        attempts = 0
        
        while len(negative_items) < num_negatives and attempts < max_attempts:
            item_id = np.random.randint(0, num_items)
            if item_id not in positive_items:
                negative_items.append(item_id)
            attempts += 1
            
        return negative_items if len(negative_items) == num_negatives else [0] * num_negatives
    
    # 构建用户-商品字典
    user_item_dict = {}
    for user_id, item_id in interactions:
        if user_id not in user_item_dict:
            user_item_dict[user_id] = set()
        user_item_dict[user_id].add(item_id)
    
    # 检查负采样质量
    test_user = list(user_item_dict.keys())[0]
    neg_samples = negative_sampling(test_user, user_item_dict, num_items, 10)
    print(f"测试用户 {test_user} 的正样本数: {len(user_item_dict[test_user])}")
    print(f"负采样结果: {neg_samples}")
    
    # 模拟一个训练批次
    batch_size = min(64, len(interactions))
    batch_indices = np.random.choice(len(interactions), batch_size, replace=False)
    
    batch_users = []
    batch_pos_items = []
    batch_neg_items = []
    
    for idx in batch_indices:
        user_id, pos_item_id = interactions[idx]
        neg_item_id = negative_sampling(user_id, user_item_dict, num_items, 1)[0]
        
        batch_users.append(user_id)
        batch_pos_items.append(pos_item_id)
        batch_neg_items.append(neg_item_id)
    
    print(f"\n批次统计:")
    print(f"批次大小: {len(batch_users)}")
    print(f"用户ID范围: {min(batch_users)} - {max(batch_users)}")
    print(f"正样本商品ID范围: {min(batch_pos_items)} - {max(batch_pos_items)}")
    print(f"负样本商品ID范围: {min(batch_neg_items)} - {max(batch_neg_items)}")
    
    # 检查是否有重复的正负样本对
    pos_neg_pairs = set(zip(batch_users, batch_pos_items, batch_neg_items))
    if len(pos_neg_pairs) < len(batch_users):
        print("❌ 警告: 发现重复的训练样本")

def diagnose_loss_issues():
    """诊断loss不动的常见原因"""
    print("\n=== Loss问题诊断 ===")
    
    common_issues = [
        "1. 学习率设置问题",
        "   - 学习率过小: loss下降极慢",
        "   - 学习率过大: loss震荡或发散",
        "   - 建议: 尝试 1e-3, 1e-4, 1e-2",
        "",
        "2. 数据质量问题", 
        "   - 数据过滤过于严格导致样本不足",
        "   - 负采样质量差",
        "   - 数据稀疏度过高",
        "",
        "3. 模型容量问题",
        "   - 嵌入维度过小",
        "   - 模型层数不合适",
        "",
        "4. 优化器问题",
        "   - 权重衰减过大",
        "   - 批次大小不合适",
        "",
        "5. 损失函数问题",
        "   - BPR损失计算有误",
        "   - 正则化系数过大",
        "",
        "6. 梯度问题",
        "   - 梯度消失",
        "   - 梯度爆炸"
    ]
    
    for issue in common_issues:
        print(issue)

def main():
    """主诊断函数"""
    print("🔍 开始诊断阿里巴巴数据集训练loss不动问题...")
    
    # 1. 分析数据预处理
    user_data, item_data = create_test_data()
    processed_data = analyze_data_preprocessing(user_data, item_data)
    
    # 2. 检查模型和损失
    if processed_data is not None:
        check_model_and_loss()
    
    # 3. 模拟训练过程
    simulate_training_process()
    
    # 4. 提供诊断建议
    diagnose_loss_issues()
    
    print("\n=== 解决建议 ===")
    print("1. 检查数据预处理: 确保过滤后仍有足够的训练样本")
    print("2. 调整学习率: 尝试 0.01, 0.001, 0.0001")  
    print("3. 减少正则化: 将weight_decay从1e-4改为1e-5或1e-6")
    print("4. 增加嵌入维度: 从64增加到128或256")
    print("5. 检查负采样: 确保负样本质量")
    print("6. 减少过滤条件: 将最小交互次数从5改为2或3")
    print("7. 增加批次大小: 尝试2048或4096")

if __name__ == "__main__":
    main()