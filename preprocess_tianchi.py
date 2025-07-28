#!/usr/bin/env python3
"""
天池移动电商推荐数据预处理脚本
按照用户提供的思路进行数据切分
"""

import pandas as pd
import numpy as np
import os

def preprocess_tianchi_data():
    """
    按照用户描述的方式处理天池数据
    """
    print("=" * 50)
    print("天池移动电商推荐数据预处理")
    print("=" * 50)
    
    # ========================
    # Step 1. 数据读取与切分
    # ========================
    print("Step 1: 数据读取与切分")
    
    # 如果没有数据文件，先创建示例数据
    data_dir = 'data/tianchi'
    user_file = os.path.join(data_dir, 'tianchi_mobile_recommend_train_user.csv')
    
    if not os.path.exists(user_file):
        print("数据文件不存在，创建示例数据...")
        create_sample_data()
    
    # 加载用户行为数据
    user_df = pd.read_csv(user_file)
    print(f"加载用户行为数据: {len(user_df)} 条记录")
    print("数据字段:", user_df.columns.tolist())
    print("数据示例:")
    print(user_df.head())
    
    # 解析时间，从时间戳中提取日期（如2014-12-18）
    user_df['date'] = user_df['time'].astype(str).str.slice(0, 8)  # 取前8位：20141218
    user_df['date'] = user_df['date'].str.replace(r'(\d{4})(\d{2})(\d{2})', r'\1-\2-\3', regex=True)
    print(f"\n时间范围: {user_df['date'].min()} 到 {user_df['date'].max()}")
    
    # 行为类型统计
    print("\n行为类型分布:")
    behavior_counts = user_df['behavior_type'].value_counts().sort_index()
    behavior_names = {1: '点击', 2: '收藏', 3: '加购物车', 4: '购买'}
    for behavior, count in behavior_counts.items():
        print(f"  {behavior}-{behavior_names.get(behavior, '未知')}: {count} 次")
    
    # 建图时用所有行为（1-点击 2-收藏 3-加购物车 4-购买）
    print("\n数据切分策略:")
    print("  - 训练集: < 2014-12-17 (所有行为类型)")
    print("  - 验证集: 2014-12-17 (购买行为)")  
    print("  - 测试集: 2014-12-18 (购买行为)")
    
    # 按照用户提供的切分方式
    train_df = user_df[user_df['date'] < '2014-12-17']
    val_df = user_df[user_df['date'] == '2014-12-17']
    test_df = user_df[user_df['date'] == '2014-12-18']
    
    print(f"\n数据切分结果:")
    print(f"  训练集: {len(train_df)} 条记录")
    print(f"  验证集: {len(val_df)} 条记录")
    print(f"  测试集: {len(test_df)} 条记录")
    
    # ========================
    # Step 2. 图构建数据分析
    # ========================
    print("\n" + "="*50)
    print("Step 2: 图构建数据分析")
    print("="*50)
    
    # 分析用户和物品数量
    unique_users = user_df['user_id'].nunique()
    unique_items = user_df['item_id'].nunique()
    unique_categories = user_df['item_category'].nunique()
    
    print(f"数据规模:")
    print(f"  用户数: {unique_users}")
    print(f"  商品数: {unique_items}")
    print(f"  类别数: {unique_categories}")
    print(f"  总交互数: {len(user_df)}")
    
    # 计算稀疏度
    sparsity = len(user_df) / (unique_users * unique_items)
    print(f"  数据稀疏度: {sparsity:.6f}")
    
    # ========================
    # Step 3. 目标任务分析
    # ========================
    print("\n" + "="*50)
    print("Step 3: 目标任务分析")
    print("="*50)
    
    # 分析购买行为（目标预测行为）
    purchase_df = user_df[user_df['behavior_type'] == 4]
    
    print("购买行为分析:")
    print(f"  总购买次数: {len(purchase_df)}")
    print(f"  购买用户数: {purchase_df['user_id'].nunique()}")
    print(f"  被购买商品数: {purchase_df['item_id'].nunique()}")
    
    # 训练集中的购买行为
    train_purchase = train_df[train_df['behavior_type'] == 4]
    val_purchase = val_df[val_df['behavior_type'] == 4]
    test_purchase = test_df[test_df['behavior_type'] == 4]
    
    print(f"\n各时间段购买行为:")
    print(f"  训练期购买: {len(train_purchase)} 次")
    print(f"  验证期购买: {len(val_purchase)} 次")
    print(f"  测试期购买: {len(test_purchase)} 次")
    
    # ========================
    # Step 4. LightGCN建图策略
    # ========================
    print("\n" + "="*50)
    print("Step 4: LightGCN建图策略")
    print("="*50)
    
    print("LightGCN建图建议:")
    print("1. 使用训练集的所有行为类型构建用户-物品交互图")
    print("2. 不同行为类型可以设置不同权重:")
    print("   - 点击(1): 权重 0.1")
    print("   - 收藏(2): 权重 0.3") 
    print("   - 加购物车(3): 权重 0.6")
    print("   - 购买(4): 权重 1.0")
    print("3. 只使用商品子集P中的物品")
    print("4. 预测目标: 用户在测试日对商品子集的购买行为")
    
    # 保存预处理后的数据
    save_processed_data(train_df, val_df, test_df)
    
    return train_df, val_df, test_df

def create_sample_data():
    """创建示例数据"""
    print("创建天池示例数据...")
    
    # 确保目录存在
    os.makedirs('data/tianchi', exist_ok=True)
    
    # 更丰富的示例数据
    np.random.seed(42)
    
    users = list(range(100000000, 100000050))  # 50个用户
    items = list(range(200000000, 200000030))  # 30个商品
    categories = [1000, 2000, 3000, 4000, 5000]  # 5个类别
    
    user_data = []
    
    # 生成2014-11-18到2014-12-18的数据
    dates = []
    for month in [11, 12]:
        if month == 11:
            for day in range(18, 31):  # 11月18-30日
                dates.append(f"2014-{month:02d}-{day:02d}")
        else:  # month == 12
            for day in range(1, 19):   # 12月1-18日
                dates.append(f"2014-{month:02d}-{day:02d}")
    
    # 为每个用户在每个日期生成行为
    for date in dates:
        for user in users:
            # 每个用户每天与部分商品交互
            user_items = np.random.choice(items, size=np.random.randint(1, 8), replace=False)
            
            for item in user_items:
                # 行为序列：点击 -> 收藏 -> 加购物车 -> 购买
                behaviors = [1]  # 至少有点击
                
                if np.random.random() > 0.7:  # 30%概率收藏
                    behaviors.append(2)
                
                if np.random.random() > 0.8:  # 20%概率加购物车
                    behaviors.append(3)
                
                if np.random.random() > 0.9:  # 10%概率购买
                    behaviors.append(4)
                
                for behavior in behaviors:
                    category = categories[item % len(categories)]
                    # 生成小时级别的时间戳
                    hour = np.random.randint(0, 24)
                    time_str = date.replace('-', '') + f"{hour:02d}"
                    user_data.append({
                        'user_id': user,
                        'item_id': item,
                        'behavior_type': behavior,
                        'user_geohash': f"geo{np.random.randint(100, 999)}",
                        'item_category': category,
                        'time': time_str
                    })
    
    # 创建用户数据
    user_df = pd.DataFrame(user_data)
    user_df.to_csv('data/tianchi/tianchi_mobile_recommend_train_user.csv', index=False)
    
    # 创建商品数据 (商品子集P)
    item_data = []
    for item in items:
        category = categories[item % len(categories)]
        item_data.append({
            'item_id': item,
            'item_geohash': f"geo{np.random.randint(100, 999)}",
            'item_category': category
        })
    
    item_df = pd.DataFrame(item_data)
    item_df.to_csv('data/tianchi/tianchi_mobile_recommend_train_item.csv', index=False)
    
    print(f"创建了 {len(user_df)} 条用户行为记录")
    print(f"创建了 {len(item_df)} 个商品记录")

def save_processed_data(train_df, val_df, test_df):
    """保存预处理后的数据"""
    output_dir = 'data/tianchi/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存切分后的数据
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # 保存购买行为数据（用于评估）
    train_purchase = train_df[train_df['behavior_type'] == 4]
    val_purchase = val_df[val_df['behavior_type'] == 4]
    test_purchase = test_df[test_df['behavior_type'] == 4]
    
    train_purchase.to_csv(os.path.join(output_dir, 'train_purchase.csv'), index=False)
    val_purchase.to_csv(os.path.join(output_dir, 'val_purchase.csv'), index=False)
    test_purchase.to_csv(os.path.join(output_dir, 'test_purchase.csv'), index=False)
    
    print(f"\n预处理数据已保存到 {output_dir}/")
    print("文件列表:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    train_df, val_df, test_df = preprocess_tianchi_data()
    
    print("\n" + "="*50)
    print("预处理完成！")
    print("接下来可以运行: python train_tianchi.py")
    print("="*50)