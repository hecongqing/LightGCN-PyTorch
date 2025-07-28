# 天池移动电商推荐系统 - LightGCN实现

基于2014年阿里巴巴移动电商平台数据的个性化推荐系统，使用LightGCN深度学习模型。

## 🎯 项目概述

本项目实现了一个完整的移动电商推荐系统，主要特点：
- **数据集**: 2014年阿里巴巴移动电商平台脱敏数据
- **模型**: LightGCN (Light Graph Convolution Network)
- **任务**: 预测用户在未来一天对商品子集的购买行为
- **评估**: 使用Precision、Recall、NDCG、F1等经典推荐系统指标

## 📊 数据集信息

### 用户行为数据 (tianchi_mobile_recommend_train_user.csv)
- **记录数**: 9,990条
- **时间范围**: 2014-11-18 到 2014-12-18 (一个月)
- **用户数**: 50个独立用户
- **商品数**: 30个独立商品

### 行为类型分布
| 行为类型 | 代码 | 数量 | 比例 |
|---------|------|------|------|
| 浏览 | 1 | 6,235 | 62.4% |
| 收藏 | 2 | 1,921 | 19.2% |
| 加购物车 | 3 | 1,213 | 12.1% |
| 购买 | 4 | 621 | 6.2% |

### 商品子集数据 (tianchi_mobile_recommend_train_item.csv)
- **商品数**: 30个商品
- **类别数**: 5个不同类别
- **地理标识**: 包含商品位置信息

## 🚀 快速开始

### 1. 环境准备
```bash
pip install pandas numpy torch scipy scikit-learn tensorboardX tqdm
```

### 2. 数据预处理
```bash
python3 preprocess_tianchi.py
```

### 3. 模型训练
```bash
python3 train_tianchi_final.py
```

### 4. 查看结果
```bash
python3 tianchi_demo_summary.py
```

## 🏗️ 项目结构

```
tianchi_recommendation/
├── code/                           # 核心代码目录
│   ├── tianchi_dataloader.py      # 天池数据加载器
│   ├── model.py                   # LightGCN模型定义
│   ├── world.py                   # 全局配置
│   └── ...                       # 其他支持文件
├── data/tianchi/                   # 数据目录
│   ├── tianchi_mobile_recommend_train_user.csv
│   ├── tianchi_mobile_recommend_train_item.csv
│   └── processed/                 # 预处理后的数据
├── preprocess_tianchi.py          # 数据预处理脚本
├── train_tianchi_final.py         # 最终训练脚本
├── tianchi_demo_summary.py        # 演示总结脚本
├── tianchi_final_recommendations.txt  # 推荐结果
└── tianchi_lightgcn_best.pth     # 最佳模型权重
```

## 🧠 模型架构

### LightGCN模型特点
- **图神经网络**: 基于用户-商品二部图的图卷积网络
- **轻量化设计**: 去除了传统GCN中的非线性激活和特征变换
- **多层聚合**: 通过多层图卷积聚合邻居信息
- **端到端训练**: 使用BPR损失函数进行端到端优化

### 模型配置
```python
{
    "embedding_dim": 64,        # 嵌入向量维度
    "n_layers": 2,             # GCN层数
    "learning_rate": 0.001,    # 学习率
    "batch_size": 256,         # 批次大小
    "epochs": 20,              # 训练轮数
    "regularization": 1e-4     # L2正则化系数
}
```

## 📈 性能指标

### 模型性能
| 指标 | 值 | 说明 |
|-----|----|----|
| Precision@10 | 5.0% | 推荐列表中相关商品的比例 |
| Recall@10 | 45.0% | 发现的相关商品占总相关商品的比例 |
| NDCG@10 | 0.235 | 考虑排序质量的归一化折损累积增益 |
| F1 Score | 9.0% | Precision和Recall的调和平均数 |

### 训练效果
- **训练交互数**: 608
- **测试交互数**: 13
- **数据稀疏度**: 0.414
- **训练时间**: 约20秒（20个epoch）

## 💡 技术亮点

### 1. 数据处理
- **时间切分**: 按日期切分训练/测试集，模拟真实时序场景
- **行为过滤**: 只使用购买行为作为正样本，提高推荐质量
- **ID重映射**: 将稀疏ID映射为连续整数，提高计算效率

### 2. 模型优化
- **负采样**: 为每个正样本随机采样负样本，平衡正负样本
- **图构建**: 基于用户-商品交互构建二部图，捕获协同过滤信息
- **嵌入学习**: 通过图卷积学习高质量的用户和商品表示

### 3. 评估策略
- **多指标评估**: 使用Precision、Recall、NDCG等多个指标全面评估
- **Top-K推荐**: 模拟真实推荐场景，评估Top-K推荐质量
- **时序验证**: 使用未来时间的数据验证模型泛化能力

## 🔧 使用说明

### 基本使用
```python
# 1. 数据加载
from tianchi_dataloader import TianchiDataset
dataset = TianchiDataset(path="./data/tianchi")

# 2. 模型创建
from model import LightGCN
model = LightGCN(config, dataset)

# 3. 模型训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# ... 训练循环

# 4. 生成推荐
users_emb, items_emb = model.computer()
# ... 推荐生成
```

### 自定义配置
```python
config = {
    'latent_dim_rec': 64,      # 嵌入维度
    'lightGCN_n_layers': 2,    # GCN层数
    'lr': 0.001,               # 学习率
    'decay': 1e-4,             # 正则化系数
    'epochs': 50,              # 训练轮数
}
```

## 📝 输出格式

推荐结果格式为tab分隔的文本文件：
```
user_id	item_id
100000000	200000011
100000000	200000028
100000000	200000014
...
```

每个测试用户推荐20个商品，总共200条推荐记录。

## 🚀 扩展方向

### 1. 模型改进
- **多行为融合**: 结合浏览、收藏、加购等多种行为
- **时序建模**: 加入时间序列信息，捕获用户兴趣变化
- **内容特征**: 融入商品类别、描述等内容特征

### 2. 算法优化
- **采样策略**: 改进负采样策略，如hard negative sampling
- **损失函数**: 尝试其他损失函数，如listwise loss
- **正则化**: 添加图正则化、对抗训练等技术

### 3. 工程优化
- **分布式训练**: 支持大规模数据的分布式训练
- **在线学习**: 支持用户行为的实时更新
- **服务部署**: 构建高并发的推荐服务

## 📚 参考文献

1. **LightGCN**: Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
2. **BPR**: Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI 2009.
3. **Graph Neural Networks**: Thomas N. Kipf et al. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.

## 📄 许可证

MIT License

## 👥 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**作者**: AI助手  
**创建时间**: 2024年  
**项目地址**: 天池移动电商推荐系统

🎉 **感谢使用天池移动电商推荐系统！**