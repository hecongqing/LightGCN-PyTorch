"""
LightGCN阿里推荐系统测试脚本
验证所有组件是否正常工作
"""

import os
import torch
import numpy as np
from alibaba_dataloader import AlibabaDataset, create_sample_data
from lightgcn_model import LightGCN, BPRLoss
from train import LightGCNTrainer
from main_alibaba import AlibabaRecommendationSystem

def test_data_loading():
    """测试数据加载功能"""
    print("=== 测试数据加载 ===")
    
    # 创建测试数据
    data_path = "./test_data"
    create_sample_data(data_path)
    
    # 加载数据集
    dataset = AlibabaDataset(
        data_path=data_path,
        behavior_types=[4],
        test_size=0.2,
        random_state=42
    )
    
    # 验证数据
    assert dataset.num_users > 0, "用户数量应该大于0"
    assert dataset.num_items > 0, "商品数量应该大于0"
    assert dataset.num_interactions > 0, "交互数量应该大于0"
    
    edge_index = dataset.get_edge_index()
    assert edge_index.shape[0] == 2, "边索引应该是2维"
    assert edge_index.shape[1] > 0, "应该有边存在"
    
    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()
    assert len(train_data) > 0, "训练数据不应为空"
    assert len(test_data) > 0, "测试数据不应为空"
    
    print("✅ 数据加载测试通过")
    return dataset

def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    
    model = LightGCN(
        num_users=100,
        num_items=50,
        embedding_dim=32,
        num_layers=2,
        dropout=0.1
    )
    
    # 验证模型参数
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "模型应该有参数"
    
    # 测试前向传播
    device = torch.device('cpu')
    model = model.to(device)
    
    # 创建示例边索引
    edge_index = torch.randint(0, 150, (2, 200))
    
    user_emb, item_emb = model(edge_index)
    assert user_emb.shape == (100, 32), f"用户嵌入形状错误: {user_emb.shape}"
    assert item_emb.shape == (50, 32), f"商品嵌入形状错误: {item_emb.shape}"
    
    # 测试预测
    user_ids = torch.randint(0, 100, (10,))
    item_ids = torch.randint(0, 50, (10,))
    scores = model.predict(user_ids, item_ids, edge_index)
    assert scores.shape == (10,), f"预测得分形状错误: {scores.shape}"
    
    print("✅ 模型创建测试通过")
    return model

def test_training():
    """测试模型训练"""
    print("\n=== 测试模型训练 ===")
    
    # 使用小数据集进行快速训练测试
    data_path = "./test_data"
    dataset = AlibabaDataset(
        data_path=data_path,
        behavior_types=[4],
        test_size=0.2,
        random_state=42
    )
    
    model = LightGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=16,  # 小维度以加快训练
        num_layers=2,
        dropout=0.1
    )
    
    device = torch.device('cpu')
    trainer = LightGCNTrainer(
        model=model,
        dataset=dataset,
        learning_rate=0.01,
        weight_decay=1e-4,
        batch_size=32,
        device=device
    )
    
    # 短时间训练
    history = trainer.train(
        num_epochs=3,
        eval_every=3,
        early_stopping_patience=5,
        save_path=None  # 不保存模型
    )
    
    assert len(history['train_losses']) > 0, "应该有训练损失记录"
    assert history['train_losses'][0] > 0, "训练损失应该大于0"
    
    print("✅ 模型训练测试通过")
    return trainer

def test_recommendation():
    """测试推荐生成"""
    print("\n=== 测试推荐生成 ===")
    
    data_path = "./test_data"
    dataset = AlibabaDataset(
        data_path=data_path,
        behavior_types=[4],
        test_size=0.2,
        random_state=42
    )
    
    model = LightGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=16,
        num_layers=2,
        dropout=0.1
    )
    
    edge_index = dataset.get_edge_index()
    
    # 测试推荐生成
    user_id = 0
    k = 5
    
    recommended_items, scores = model.recommend(
        user_id=user_id,
        edge_index=edge_index,
        k=k
    )
    
    assert len(recommended_items) == k, f"推荐商品数量错误: {len(recommended_items)}"
    assert len(scores) == k, f"推荐得分数量错误: {len(scores)}"
    assert all(0 <= item < dataset.num_items for item in recommended_items), "推荐商品ID超出范围"
    
    print("✅ 推荐生成测试通过")

def test_evaluation():
    """测试模型评估"""
    print("\n=== 测试模型评估 ===")
    
    # 创建简单的测试数据
    data_path = "./test_data"
    dataset = AlibabaDataset(
        data_path=data_path,
        behavior_types=[4],
        test_size=0.2,
        random_state=42
    )
    
    model = LightGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=16,
        num_layers=2,
        dropout=0.1
    )
    
    device = torch.device('cpu')
    trainer = LightGCNTrainer(
        model=model,
        dataset=dataset,
        learning_rate=0.01,
        weight_decay=1e-4,
        batch_size=32,
        device=device
    )
    
    # 快速训练
    trainer.train(num_epochs=2, eval_every=2)
    
    # 评估模型
    metrics = trainer.evaluate(k_list=[5, 10])
    
    required_metrics = ['Precision@5', 'Recall@5', 'NDCG@5', 
                       'Precision@10', 'Recall@10', 'NDCG@10']
    
    for metric in required_metrics:
        assert metric in metrics, f"缺少评估指标: {metric}"
        assert 0 <= metrics[metric] <= 1, f"指标 {metric} 超出范围: {metrics[metric]}"
    
    print("✅ 模型评估测试通过")

def test_complete_system():
    """测试完整推荐系统"""
    print("\n=== 测试完整推荐系统 ===")
    
    data_path = "./test_data"
    device = torch.device('cpu')
    
    # 创建推荐系统
    system = AlibabaRecommendationSystem(
        data_path=data_path,
        device=device
    )
    
    # 加载数据
    dataset = system.load_data(behavior_types=[4], test_size=0.2)
    
    # 构建模型
    model = system.build_model()
    
    # 训练模型（短时间）
    training_config = {
        'num_epochs': 3,
        'batch_size': 32,
        'learning_rate': 0.01,
        'eval_every': 3
    }
    
    history = system.train_model(training_config, save_path=None)
    
    # 生成推荐
    user_id = 0
    recommendations = system.generate_recommendations(user_id, k=5)
    assert len(recommendations) == 5, "推荐数量不正确"
    
    # 批量推荐
    user_ids = [0, 1, 2]
    batch_recommendations = system.batch_recommend(user_ids, k=3)
    assert len(batch_recommendations) == 3, "批量推荐用户数量不正确"
    
    # 评估系统
    metrics = system.evaluate_system([5, 10])
    assert len(metrics) > 0, "评估指标为空"
    
    print("✅ 完整推荐系统测试通过")

def test_bpr_loss():
    """测试BPR损失函数"""
    print("\n=== 测试BPR损失函数 ===")
    
    model = LightGCN(
        num_users=100,
        num_items=50,
        embedding_dim=16,
        num_layers=2
    )
    
    criterion = BPRLoss(lambda_reg=1e-4)
    
    # 创建示例得分
    pos_scores = torch.randn(32)
    neg_scores = torch.randn(32)
    
    loss = criterion(pos_scores, neg_scores, model)
    
    assert loss.item() > 0, "BPR损失应该大于0"
    assert torch.isfinite(loss), "BPR损失应该是有限数"
    
    print("✅ BPR损失函数测试通过")

def cleanup():
    """清理测试文件"""
    print("\n=== 清理测试文件 ===")
    
    import shutil
    test_dirs = ["./test_data", "./models", "./results"]
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"删除目录: {dir_path}")

def main():
    """主测试函数"""
    print("🧪 开始LightGCN阿里推荐系统测试")
    print("=" * 50)
    
    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 运行所有测试
        test_data_loading()
        test_model_creation()
        test_bpr_loss()
        test_training()
        test_recommendation()
        test_evaluation()
        test_complete_system()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！系统工作正常。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试文件
        cleanup()
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)