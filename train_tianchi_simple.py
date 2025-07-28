#!/usr/bin/env python3
"""
天池移动电商推荐系统训练脚本 - 简化版
使用LightGCN模型
"""

import os
import sys
import pandas as pd
import numpy as np
import torch

# 添加代码路径
sys.path.append('./code')

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    """主函数"""
    print("Starting Tianchi Mobile E-commerce Recommendation Training")
    print("=" * 60)
    
    # 模拟命令行参数
    sys.argv = [
        'train_tianchi_simple.py',
        '--dataset=tianchi',
        '--model=lgn',
        '--epochs=10',  # 减少epoch用于测试
        '--lr=0.001',
        '--decay=1e-4',
        '--layer=2',    # 减少层数用于测试
        '--recdim=32',  # 减少维度用于测试
        '--bpr_batch=256',  # 减少batch size
        '--testbatch=100',
        '--topks=[10,20]',
        '--seed=2020',
        '--comment=tianchi_simple'
    ]
    
    # 导入world和其他模块
    import world
    import utils
    from world import cprint
    import torch
    import numpy as np
    import time
    import Procedure
    from os.path import join
    
    # 确保dataset设置为tianchi
    world.dataset = 'tianchi'
    
    # 设置随机种子
    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)
    
    # 创建天池数据加载器
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
    
    # 创建损失函数
    bpr = utils.BPRLoss(Recmodel, world.config)
    
    # 训练参数
    epochs = world.config.get('epochs', 10)  # 默认10个epoch
    
    print("Start Training!")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {world.config['lr']}")
    print(f"Batch size: {world.config['bpr_batch_size']}")
    print(f"Embedding dimension: {world.config['latent_dim_rec']}")
    print(f"Layers: {world.config['lightGCN_n_layers']}")
    print("-" * 50)
    
    # 训练循环
    best_f1 = 0.0
    for epoch in range(epochs):
        start = time.time()
        
        # 创建tensorboard writer
        from tensorboardX import SummaryWriter
        w = SummaryWriter(log_dir=f'./logs/tianchi_simple_epoch_{epoch}')
        
        # 训练
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=1, w=w)
        
        # 关闭writer
        w.close()
        
        # 测试
        if epoch % 5 == 0 or epoch == epochs - 1:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w=None, multicore=0)
            
            # 获取F1值 (precision + recall的调和平均)
            precision = results['precision'][0]  # top-10 precision
            recall = results['recall'][0]        # top-10 recall
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            if f1 > best_f1:
                best_f1 = f1
                print(f"New best F1: {best_f1:.6f}")
                
                # 保存模型
                torch.save(Recmodel.state_dict(), f'tianchi_lightgcn_best.pth')
        
        end = time.time()
        print(f"Epoch {epoch:3d} | Time: {end-start:.2f}s | {output_information}")
    
    print(f"\nTraining completed! Best F1: {best_f1:.6f}")
    
    # 生成推荐结果
    print("\nGenerating recommendations...")
    generate_recommendations(dataset, Recmodel, top_k=20)

def generate_recommendations(dataset, model, top_k=20):
    """生成推荐结果"""
    model.eval()
    
    with torch.no_grad():
        # 获取所有用户和物品的嵌入
        users_emb, items_emb = model.getUsersRating()
        
        # 生成推荐
        recommendations = []
        test_users = list(dataset.testDict.keys())
        
        print(f"Generating recommendations for {len(test_users)} test users...")
        
        for i, user in enumerate(test_users):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(test_users)}")
                
            # 获取用户已购买的物品
            pos_items = set(dataset.allPos[user])
            
            # 计算用户与所有物品的得分
            user_emb = users_emb[user].unsqueeze(0)  # [1, dim]
            scores = torch.matmul(user_emb, items_emb.transpose(0, 1)).squeeze()  # [n_items]
            
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
    output_file = 'tianchi_recommendations.txt'
    with open(output_file, 'w') as f:
        for user_id, item_id in recommendations:
            f.write(f"{user_id}\t{item_id}\n")
    
    print(f"Saved {len(recommendations)} recommendations to {output_file}")
    print("Sample recommendations:")
    for i, (user_id, item_id) in enumerate(recommendations[:10]):
        print(f"  {user_id}\t{item_id}")

if __name__ == "__main__":
    main()