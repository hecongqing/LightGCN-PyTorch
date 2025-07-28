#!/usr/bin/env python3
"""
天池移动电商推荐系统训练脚本
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

import world
import utils
from world import cprint
import Procedure
import time
from tensorboardX import SummaryWriter
from os.path import join

def create_tianchi_data():
    """创建更多的天池示例数据"""
    print("Creating tianchi sample data...")
    
    # 创建更多的示例数据
    user_data = []
    item_data = []
    
    # 生成用户行为数据
    users = [100017697, 1100025197, 1100030370, 100062756, 1100064788, 
             100073680, 100077356, 100080123, 100085467, 100090234]
    items = [273020077, 92334510, 29240499, 130930712, 16234722,
             316904196, 323228373, 333757713, 347872201, 338630082]
    categories = [1832, 4076, 982]
    behaviors = [1, 2, 3, 4]  # 1-点击 2-收藏 3-加购物车 4-购买
    
    # 生成训练数据 (2014-11-18 到 2014-12-16)
    dates = ['2014111817', '2014111818', '2014111919', '2014112017', '2014112118',
             '2014112819', '2014120117', '2014120518', '2014121017', '2014121117',
             '2014121217', '2014121317', '2014121417', '2014121517', '2014121617']
    
    for date in dates:
        for user in users:
            for item in items[:7]:  # 每个用户对前7个商品有行为
                for behavior in behaviors:
                    if np.random.random() > 0.7:  # 30%的概率产生行为
                        category = categories[item % 3]
                        user_data.append({
                            'user_id': user,
                            'item_id': item,
                            'behavior_type': behavior,
                            'user_geohash': '97lk14c',
                            'item_category': category,
                            'time': date
                        })
    
    # 生成测试数据 (2014-12-18)
    test_date = '2014121817'
    for user in users[:8]:  # 前8个用户在测试日有购买行为
        for item in items[:5]:  # 对前5个商品有购买行为
            if np.random.random() > 0.8:  # 20%的概率购买
                category = categories[item % 3]
                user_data.append({
                    'user_id': user,
                    'item_id': item,
                    'behavior_type': 4,  # 购买行为
                    'user_geohash': '97lk14c',
                    'item_category': category,
                    'time': test_date
                })
    
    # 创建商品数据
    for item in items:
        category = categories[item % 3]
        item_data.append({
            'item_id': item,
            'item_geohash': '97lk14c',
            'item_category': category
        })
    
    # 保存数据
    user_df = pd.DataFrame(user_data)
    item_df = pd.DataFrame(item_data)
    
    user_df.to_csv('data/tianchi/tianchi_mobile_recommend_train_user.csv', index=False)
    item_df.to_csv('data/tianchi/tianchi_mobile_recommend_train_item.csv', index=False)
    
    print(f"Created {len(user_df)} user interactions and {len(item_df)} items")
    print("Sample user data:")
    print(user_df.head())
    print("\nSample item data:")
    print(item_df.head())

def train_tianchi_lightgcn():
    """训练天池数据集的LightGCN模型"""
    
    # 确保数据存在
    if not os.path.exists('data/tianchi/tianchi_mobile_recommend_train_user.csv'):
        create_tianchi_data()
    
    # 设置数据集为tianchi  
    world.dataset = 'tianchi'
    
    # 设置随机种子
    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)
    
    # 导入数据集和模型
    import register
    from register import dataset
    
    print(f"Dataset info:")
    print(f"  Users: {dataset.n_users}")
    print(f"  Items: {dataset.m_items}")
    print(f"  Train interactions: {dataset.trainDataSize}")
    print(f"  Test interactions: {dataset.testDataSize}")
    
    # 创建模型
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)
    
    # 设置权重文件路径
    weight_file = utils.getFileName()
    print(f"Load and save to {weight_file}")
    
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            world.cprint(f"Loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    
    Neg_k = 1
    
    # 初始化tensorboard
    if world.tensorboard:
        w = SummaryWriter(
            join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-tianchi-" + world.comment)
        )
    else:
        w = None
        world.cprint("Not enable tensorboard")
    
    try:
        print("Starting training...")
        for epoch in range(world.TRAIN_epochs):
            start = time.time()
            
            # 每10个epoch测试一次
            if epoch % 10 == 0:
                cprint(f"[TEST] Epoch {epoch}")
                Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            
            # 训练一个epoch
            output_information = Procedure.BPR_train_original(
                dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
            )
            
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            
            # 保存模型
            torch.save(Recmodel.state_dict(), weight_file)
            
            # 早停检查
            if epoch > 50 and epoch % 50 == 0:
                print("Performing final test...")
                results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                print(f"Results at epoch {epoch}: {results}")
                
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        if world.tensorboard:
            w.close()
        
        # 最终测试
        print("Performing final test...")
        final_results = Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, None, world.config['multicore'])
        print(f"Final results: {final_results}")
        
        # 生成推荐结果
        generate_recommendations(dataset, Recmodel)

def generate_recommendations(dataset, model, top_k=20):
    """生成推荐结果并保存"""
    print("Generating recommendations...")
    
    model.eval()
    recommendations = []
    
    with torch.no_grad():
        # 获取所有用户嵌入和物品嵌入
        users_emb, items_emb = model.computer()
        
        # 为每个测试用户生成推荐
        test_users = list(dataset.testDict.keys())
        
        for user in test_users:
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
    # 设置模拟的命令行参数
    import sys
    
    # 模拟命令行参数
    sys.argv = [
        'train_tianchi.py',
        '--dataset=tianchi',
        '--model=lgn',
        '--epochs=100',
        '--lr=0.001',
        '--decay=1e-4',
        '--layer=3',
        '--recdim=64',
        '--bpr_batch=1024',
        '--testbatch=100',
        '--topks=[10,20]',
        '--seed=2020',
        '--comment=tianchi_demo'
    ]
    
    print("Starting Tianchi Mobile E-commerce Recommendation Training")
    print("=" * 60)
    
    train_tianchi_lightgcn()