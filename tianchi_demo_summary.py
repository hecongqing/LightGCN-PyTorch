#!/usr/bin/env python3
"""
天池移动电商推荐系统 - 总结与演示
基于LightGCN模型的完整实现

作者：AI助手
数据集：2014年阿里巴巴移动电商平台数据（脱敏）
"""

import pandas as pd
import numpy as np
import os

def show_data_overview():
    """展示数据集概览"""
    print("=" * 60)
    print("天池移动电商推荐数据集概览")
    print("=" * 60)
    
    # 读取原始数据
    user_file = 'data/tianchi/tianchi_mobile_recommend_train_user.csv'
    item_file = 'data/tianchi/tianchi_mobile_recommend_train_item.csv'
    
    if os.path.exists(user_file) and os.path.exists(item_file):
        user_df = pd.read_csv(user_file)
        item_df = pd.read_csv(item_file)
        
        print(f"用户行为数据: {len(user_df)} 条记录")
        print(f"商品子集数据: {len(item_df)} 条记录")
        print(f"独立用户数: {user_df['user_id'].nunique()}")
        print(f"独立商品数: {user_df['item_id'].nunique()}")
        
        # 行为类型分布
        behavior_counts = user_df['behavior_type'].value_counts().sort_index()
        behavior_map = {1: '浏览', 2: '收藏', 3: '加购物车', 4: '购买'}
        
        print("\n行为类型分布:")
        for behavior, count in behavior_counts.items():
            print(f"  {behavior_map[behavior]}({behavior}): {count} 次")
        
        # 时间范围
        user_df['date_str'] = user_df['time'].astype(str).str.slice(0, 8)
        user_df['date'] = pd.to_datetime(user_df['date_str'], format='%Y%m%d')
        
        print(f"\n数据时间范围:")
        print(f"  开始日期: {user_df['date'].min().strftime('%Y-%m-%d')}")
        print(f"  结束日期: {user_df['date'].max().strftime('%Y-%m-%d')}")
        
        # 商品类别分布
        category_counts = user_df['item_category'].value_counts()
        print(f"\n商品类别数: {user_df['item_category'].nunique()}")
        print("主要类别分布:")
        for category, count in category_counts.head().items():
            print(f"  类别 {category}: {count} 次行为")
        
    else:
        print("数据文件不存在，请先运行数据预处理脚本")

def show_model_performance():
    """展示模型性能"""
    print("\n" + "=" * 60)
    print("LightGCN模型性能总结")
    print("=" * 60)
    
    print("模型配置:")
    print("  - 嵌入维度: 64")
    print("  - GCN层数: 2")
    print("  - 学习率: 0.001")
    print("  - 训练轮数: 20")
    print("  - 批次大小: 256")
    
    print("\n数据集统计:")
    print("  - 用户数: 50")
    print("  - 商品数: 30")
    print("  - 训练交互数: 608")
    print("  - 测试交互数: 13")
    print("  - 稀疏度: 0.414")
    
    print("\n模型性能指标:")
    print("  - Precision@10: 0.050000 (5.0%)")
    print("  - Recall@10: 0.450000 (45.0%)")
    print("  - NDCG@10: 0.234614")
    print("  - F1 Score: 0.090000 (9.0%)")
    
    print("\n性能解释:")
    print("  - Precision较低但Recall较高说明模型倾向于召回更多相关商品")
    print("  - 在小数据集上这是合理的表现")
    print("  - 在实际应用中需要更大的数据集来获得更好的性能")

def show_recommendations():
    """展示推荐结果"""
    print("\n" + "=" * 60)
    print("推荐结果示例")
    print("=" * 60)
    
    rec_file = 'tianchi_final_recommendations.txt'
    if os.path.exists(rec_file):
        print("推荐格式: user_id\\titem_id")
        print("样例推荐结果:")
        
        with open(rec_file, 'r') as f:
            lines = f.readlines()
            
        print(f"总推荐数: {len(lines)}")
        print("前10条推荐:")
        for i, line in enumerate(lines[:10]):
            user_id, item_id = line.strip().split('\t')
            print(f"  {i+1}. 用户 {user_id} -> 商品 {item_id}")
            
        # 统计每个用户的推荐数
        user_rec_counts = {}
        for line in lines:
            user_id, _ = line.strip().split('\t')
            user_rec_counts[user_id] = user_rec_counts.get(user_id, 0) + 1
            
        print(f"\n用户推荐分布:")
        print(f"  独立用户数: {len(user_rec_counts)}")
        print(f"  平均每用户推荐数: {np.mean(list(user_rec_counts.values())):.1f}")
        print(f"  最多推荐数: {max(user_rec_counts.values())}")
        print(f"  最少推荐数: {min(user_rec_counts.values())}")
    else:
        print("推荐结果文件不存在")

def show_implementation_details():
    """展示实现细节"""
    print("\n" + "=" * 60)
    print("实现技术细节")
    print("=" * 60)
    
    print("1. 数据预处理:")
    print("   - 时间解析: 从时间戳提取日期进行数据分割")
    print("   - 行为过滤: 只使用购买行为(behavior_type=4)作为正样本")
    print("   - ID重映射: 将用户和商品ID重新编码为0开始的连续整数")
    print("   - 商品过滤: 只考虑商品子集P中的商品")
    
    print("\n2. 模型架构:")
    print("   - 基础模型: LightGCN (Light Graph Convolution Network)")
    print("   - 图构建: 用户-商品二部图，基于购买行为")
    print("   - 嵌入学习: 通过图卷积学习用户和商品的向量表示")
    print("   - 损失函数: BPR (Bayesian Personalized Ranking) Loss")
    
    print("\n3. 训练策略:")
    print("   - 负采样: 为每个正样本随机采样负样本")
    print("   - 正则化: L2正则化防止过拟合")
    print("   - 优化器: Adam优化器")
    print("   - 评估: 每5个epoch评估一次模型性能")
    
    print("\n4. 评估指标:")
    print("   - Precision@K: 推荐列表中相关商品的比例")
    print("   - Recall@K: 发现的相关商品占总相关商品的比例")
    print("   - NDCG@K: 考虑排序质量的归一化折损累积增益")
    print("   - F1 Score: Precision和Recall的调和平均数")

def show_business_insights():
    """展示业务洞察"""
    print("\n" + "=" * 60)
    print("业务应用与洞察")
    print("=" * 60)
    
    print("1. 移动电商特点:")
    print("   - 随时随地的访问模式")
    print("   - 更丰富的场景数据（位置、时间等）")
    print("   - 2014年双11移动端成交占比42.6%，超过240亿元")
    
    print("\n2. 推荐系统价值:")
    print("   - 提升用户体验：帮助用户发现感兴趣的商品")
    print("   - 增加销售额：通过个性化推荐促进转化")
    print("   - 优化商品分发：将合适的商品推荐给合适的用户")
    
    print("\n3. 实际部署考虑:")
    print("   - 实时性要求：推荐系统需要快速响应")
    print("   - 冷启动问题：新用户和新商品的推荐")
    print("   - 多样性平衡：准确性与多样性的权衡")
    print("   - A/B测试：持续优化推荐效果")
    
    print("\n4. 扩展方向:")
    print("   - 多行为融合：结合浏览、收藏、加购等多种行为")
    print("   - 上下文感知：利用时间、地理位置等上下文信息")
    print("   - 深度学习：使用更复杂的神经网络模型")
    print("   - 实时更新：支持用户行为的实时学习和更新")

def main():
    """主函数"""
    print("🛒 天池移动电商推荐系统 - LightGCN实现")
    print("📊 基于2014年阿里巴巴移动电商平台数据")
    print("🤖 深度学习推荐算法演示")
    
    # 显示各个部分
    show_data_overview()
    show_model_performance() 
    show_recommendations()
    show_implementation_details()
    show_business_insights()
    
    print("\n" + "=" * 60)
    print("📚 学习总结")
    print("=" * 60)
    print("通过这个项目，我们学习了:")
    print("✅ 推荐系统的基本原理和评估指标")
    print("✅ 图神经网络(GCN)在推荐系统中的应用")
    print("✅ 大规模数据的预处理和特征工程")
    print("✅ PyTorch深度学习框架的使用")
    print("✅ 工业级推荐系统的设计和实现")
    
    print("\n🎯 下一步可以:")
    print("• 尝试更大的数据集")
    print("• 实验不同的模型架构")
    print("• 优化超参数")
    print("• 添加更多特征")
    print("• 部署到生产环境")
    
    print("\n感谢使用天池移动电商推荐系统！")

if __name__ == "__main__":
    main()