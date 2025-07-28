#!/usr/bin/env python3
"""
天池移动电商推荐系统训练脚本 - 最终版本
使用LightGCN模型，简化版本避免复杂依赖
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import time

# 添加代码路径
sys.path.append('./code')

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def simple_test(dataset, model, top_k=10):
    """简化的测试函数，不使用tensorboard"""
    model.eval()
    
    with torch.no_grad():
        # 获取所有用户和物品的嵌入
        users_emb, items_emb = model.computer()
        
        # 测试用户
        test_users = list(dataset.testDict.keys())
        
        if len(test_users) == 0:
            print("No test users found!")
            return {'precision': [0.0], 'recall': [0.0], 'ndcg': [0.0]}
        
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        valid_users = 0
        
        for user in test_users:
            # 获取用户的真实正样本
            ground_truth = set(dataset.testDict[user])
            if len(ground_truth) == 0:
                continue
                
            # 获取用户已购买的物品（训练集）
            pos_items = set(dataset.allPos[user])
            
            # 计算用户与所有物品的得分
            user_emb = users_emb[user].unsqueeze(0)
            scores = torch.matmul(user_emb, items_emb.transpose(0, 1)).squeeze()
            
            # 排除已购买的物品
            for item in pos_items:
                scores[item] = -float('inf')
            
            # 获取top-k推荐
            _, top_items = torch.topk(scores, min(top_k, dataset.m_items))
            recommended_items = set(top_items.cpu().numpy())
            
            # 计算指标
            hits = len(recommended_items & ground_truth)
            precision = hits / len(recommended_items) if len(recommended_items) > 0 else 0.0
            recall = hits / len(ground_truth) if len(ground_truth) > 0 else 0.0
            
            # 简化的NDCG计算
            dcg = 0.0
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), top_k)))
            
            for i, item in enumerate(top_items.cpu().numpy()):
                if item in ground_truth:
                    dcg += 1 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            precision_sum += precision
            recall_sum += recall
            ndcg_sum += ndcg
            valid_users += 1
        
        if valid_users == 0:
            return {'precision': [0.0], 'recall': [0.0], 'ndcg': [0.0]}
        
        avg_precision = precision_sum / valid_users
        avg_recall = recall_sum / valid_users
        avg_ndcg = ndcg_sum / valid_users
        
        print(f"Precision@{top_k}: {avg_precision:.6f}")
        print(f"Recall@{top_k}: {avg_recall:.6f}")
        print(f"NDCG@{top_k}: {avg_ndcg:.6f}")
        
        return {
            'precision': [avg_precision],
            'recall': [avg_recall], 
            'ndcg': [avg_ndcg]
        }

def simple_bpr_train(dataset, model, optimizer, epoch):
    """简化的BPR训练函数"""
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    # 从训练数据中采样
    train_users = dataset.trainUniqueUsers
    batch_size = 256
    
    for start_idx in range(0, len(train_users), batch_size):
        batch_users = train_users[start_idx:start_idx + batch_size]
        
        # 为每个用户采样正负样本
        pos_items = []
        neg_items = []
        users_batch = []
        
        for user in batch_users:
            # 获取用户的正样本
            user_pos_items = dataset.allPos[user]
            if len(user_pos_items) == 0:
                continue
                
            # 随机选择一个正样本
            pos_item = np.random.choice(user_pos_items)
            
            # 随机选择一个负样本
            while True:
                neg_item = np.random.randint(0, dataset.m_items)
                if neg_item not in user_pos_items:
                    break
            
            users_batch.append(user)
            pos_items.append(pos_item)
            neg_items.append(neg_item)
        
        if len(users_batch) == 0:
            continue
            
        # 转换为tensor
        import world
        users_tensor = torch.LongTensor(users_batch).to(world.device)
        pos_tensor = torch.LongTensor(pos_items).to(world.device)
        neg_tensor = torch.LongTensor(neg_items).to(world.device)
        
        # 前向传播 - 使用getEmbedding获取嵌入
        users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego = model.getEmbedding(users_tensor, pos_tensor, neg_tensor)
        
        # BPR损失
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # L2正则化
        reg_loss = 0.0001 * (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / len(users_batch)
        
        loss = bpr_loss + reg_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return f"BPR Loss: {avg_loss:.6f}"

def main():
    """主函数"""
    print("Starting Tianchi Mobile E-commerce Recommendation Training")
    print("=" * 60)
    
    # 模拟命令行参数
    sys.argv = [
        'train_tianchi_final.py',
        '--dataset=tianchi',
        '--model=lgn',
        '--epochs=20',
        '--lr=0.001',
        '--decay=1e-4',
        '--layer=2',
        '--recdim=64',
        '--bpr_batch=256',
        '--testbatch=10',  # 减小测试batch size
        '--topks=[10,20]',
        '--seed=2020',
        '--comment=tianchi_final'
    ]
    
    # 导入必要模块
    import world
    import utils
    from world import cprint
    
    # 设置参数
    world.dataset = 'tianchi'
    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)
    
    # 创建数据集
    from tianchi_dataloader import TianchiDataset
    dataset = TianchiDataset(path="./data/tianchi")
    
    print(f"Dataset info:")
    print(f"  Users: {dataset.n_users}")
    print(f"  Items: {dataset.m_items}")
    print(f"  Train interactions: {dataset.trainDataSize}")
    print(f"  Test interactions: {dataset.testDataSize}")
    print(f"  Sparsity: {(dataset.trainDataSize + dataset.testDataSize) / dataset.n_users / dataset.m_items:.6f}")
    
    # 创建模型
    import model
    Recmodel = model.LightGCN(world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(Recmodel.parameters(), lr=world.config['lr'])
    
    # 训练参数
    epochs = 20
    
    print("Start Training!")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {world.config['lr']}")
    print(f"Embedding dimension: {world.config['latent_dim_rec']}")
    print(f"Layers: {world.config['lightGCN_n_layers']}")
    print("-" * 50)
    
    # 训练循环
    best_f1 = 0.0
    for epoch in range(epochs):
        start = time.time()
        
        # 训练
        train_info = simple_bpr_train(dataset, Recmodel, optimizer, epoch)
        
        # 测试
        if epoch % 5 == 0 or epoch == epochs - 1:
            cprint("[TEST]")
            results = simple_test(dataset, Recmodel, top_k=10)
            
            # 计算F1值
            precision = results['precision'][0]
            recall = results['recall'][0]
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            if f1 > best_f1:
                best_f1 = f1
                print(f"New best F1: {best_f1:.6f}")
                torch.save(Recmodel.state_dict(), 'tianchi_lightgcn_best.pth')
        
        end = time.time()
        print(f"Epoch {epoch:3d} | Time: {end-start:.2f}s | {train_info}")
    
    print(f"\nTraining completed! Best F1: {best_f1:.6f}")
    
    # 生成最终推荐
    print("\nGenerating final recommendations...")
    generate_recommendations(dataset, Recmodel)

def generate_recommendations(dataset, model, top_k=20):
    """生成推荐结果，符合比赛格式"""
    model.eval()
    
    with torch.no_grad():
        users_emb, items_emb = model.computer()
        
        recommendations = []
        test_users = list(dataset.testDict.keys())
        
        print(f"Generating recommendations for {len(test_users)} test users...")
        
        for user in test_users:
            # 获取用户已购买的物品
            pos_items = set(dataset.allPos[user])
            
            # 计算得分
            user_emb = users_emb[user].unsqueeze(0)
            scores = torch.matmul(user_emb, items_emb.transpose(0, 1)).squeeze()
            
            # 排除已购买的物品
            for item in pos_items:
                scores[item] = -float('inf')
            
            # 获取top-k推荐
            _, top_items = torch.topk(scores, min(top_k, dataset.m_items))
            
            # 转换为原始ID
            original_user_id = dataset.reverse_user_id_map[user]
            
            for item in top_items.cpu().numpy():
                if item in dataset.reverse_item_id_map:
                    original_item_id = dataset.reverse_item_id_map[item]
                    recommendations.append((original_user_id, original_item_id))
    
    # 保存推荐结果
    output_file = 'tianchi_final_recommendations.txt'
    with open(output_file, 'w') as f:
        for user_id, item_id in recommendations:
            f.write(f"{user_id}\t{item_id}\n")
    
    print(f"Saved {len(recommendations)} recommendations to {output_file}")
    print("Sample recommendations (user_id\\titem_id):")
    for i, (user_id, item_id) in enumerate(recommendations[:10]):
        print(f"  {user_id}\t{item_id}")
    
    return recommendations

if __name__ == "__main__":
    main()