# LightGCN阿里移动推荐系统

基于PyTorch Geometric实现的LightGCN推荐系统，使用阿里移动推荐算法挑战赛数据集进行教学演示。

## 🎯 项目简介

本项目实现了一个完整的基于图神经网络的推荐系统，主要特点：

- 🔥 **LightGCN模型**: 轻量级图卷积网络，专为推荐系统优化
- 📊 **阿里数据集**: 基于真实的移动电商数据
- 🚀 **完整流程**: 从数据处理到模型部署的端到端实现
- 📚 **教学友好**: 详细的中文注释和Jupyter教程
- 🛠 **易于扩展**: 模块化设计，便于定制和改进

## 📁 项目结构

```
├── alibaba_dataloader.py      # 阿里数据集加载器
├── lightgcn_model.py          # LightGCN模型实现
├── train.py                   # 模型训练脚本
├── main_alibaba.py            # 主运行程序
├── requirements_alibaba.txt   # 依赖包列表
├── LightGCN_Alibaba_Tutorial.ipynb  # 教学笔记本
└── README_LightGCN_Alibaba.md # 项目说明文档
```

## 🔧 环境安装

### 1. 安装依赖包

```bash
pip install -r requirements_alibaba.txt
```

主要依赖：
- `torch>=1.8.0`
- `torch-geometric>=2.0.0`
- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.4.0`

### 2. 数据准备

#### 使用真实阿里数据集
1. 从[阿里移动推荐挑战赛](https://tianchi.aliyun.com/competition/entrance/1/information)下载数据
2. 将数据文件放入 `./alibaba_data/` 目录
3. 确保文件名为：
   - `tianchi_mobile_recommend_train_user.zip` 或 `tianchi_mobile_recommend_train_user.csv`
   - `tianchi_mobile_recommend_train_item.csv`

#### 使用示例数据
如果没有真实数据，程序会自动生成示例数据用于测试。

## 🚀 快速开始

### 1. 训练模型

```bash
# 基础训练
python main_alibaba.py --mode train

# 自定义参数训练
python main_alibaba.py --mode train \
    --embedding_dim 128 \
    --num_layers 4 \
    --num_epochs 200 \
    --batch_size 2048 \
    --learning_rate 0.001
```

### 2. 生成推荐

```bash
# 为指定用户生成推荐
python main_alibaba.py --mode recommend --user_id 0 --k 10

# 为所有用户生成推荐
python main_alibaba.py --mode recommend --user_id -1 --k 10
```

### 3. 评估模型

```bash
python main_alibaba.py --mode evaluate --model_path ./models/lightgcn_alibaba.pth
```

## 📖 教学教程

### Jupyter笔记本教程

运行 `LightGCN_Alibaba_Tutorial.ipynb` 获得完整的交互式教学体验：

```bash
jupyter notebook LightGCN_Alibaba_Tutorial.ipynb
```

教程内容包括：
1. 📚 **理论基础**: 推荐系统和图神经网络原理
2. 📊 **数据分析**: 阿里数据集的探索性分析
3. 🔧 **模型实现**: LightGCN的详细实现过程
4. 🏋️ **模型训练**: BPR损失函数和训练策略
5. 📈 **结果分析**: 评估指标和可视化分析
6. 🔍 **模型解释**: 嵌入空间可视化和相似性分析

## 🏗 核心组件详解

### 1. 数据加载器 (`alibaba_dataloader.py`)

- **功能**: 处理阿里移动推荐数据集
- **特性**: 
  - 支持多种行为类型过滤
  - 自动构建用户-商品二部图
  - 负采样和数据划分
  - 图结构的边索引生成

```python
from alibaba_dataloader import AlibabaDataset

dataset = AlibabaDataset(
    data_path="./alibaba_data",
    behavior_types=[4],  # 只考虑购买行为
    test_size=0.2
)
```

### 2. LightGCN模型 (`lightgcn_model.py`)

- **功能**: 轻量级图卷积网络实现
- **特性**:
  - 简化的图卷积操作
  - 多层嵌入聚合
  - BPR损失函数
  - 高效的推荐生成

```python
from lightgcn_model import LightGCN

model = LightGCN(
    num_users=1000,
    num_items=500,
    embedding_dim=64,
    num_layers=3
)
```

### 3. 训练器 (`train.py`)

- **功能**: 完整的模型训练流程
- **特性**:
  - 早停机制
  - 训练曲线可视化
  - 模型保存和加载
  - 多指标评估

```python
from train import LightGCNTrainer

trainer = LightGCNTrainer(model, dataset, device='cuda')
history = trainer.train(num_epochs=100)
```

### 4. 推荐系统 (`main_alibaba.py`)

- **功能**: 端到端的推荐系统
- **特性**:
  - 命令行接口
  - 批量推荐生成
  - 结果分析和导出
  - 多种运行模式

## 📊 模型性能

### 评估指标

- **Precision@K**: 推荐精度
- **Recall@K**: 推荐召回率  
- **NDCG@K**: 归一化折扣累积增益

### 典型结果（示例数据）

| 指标 | @5 | @10 | @20 |
|------|----|----|-----|
| Precision | 0.054 | 0.042 | 0.031 |
| Recall | 0.089 | 0.156 | 0.234 |
| NDCG | 0.072 | 0.098 | 0.127 |

## 🔧 参数说明

### 模型参数

- `embedding_dim`: 嵌入维度 (默认: 64)
- `num_layers`: GCN层数 (默认: 3)
- `dropout`: Dropout比例 (默认: 0.1)

### 训练参数

- `learning_rate`: 学习率 (默认: 0.001)
- `weight_decay`: 权重衰减 (默认: 1e-4)
- `batch_size`: 批次大小 (默认: 1024)
- `num_epochs`: 训练轮数 (默认: 100)

### 数据参数

- `behavior_types`: 行为类型 (默认: [4] 只考虑购买)
- `test_size`: 测试集比例 (默认: 0.2)

## 🚀 高级用法

### 1. 自定义数据集

继承 `AlibabaDataset` 类来处理自定义数据：

```python
class CustomDataset(AlibabaDataset):
    def _load_data(self):
        # 实现自定义数据加载逻辑
        pass
```

### 2. 模型扩展

在 `lightgcn_model.py` 基础上添加新特性：

```python
class EnhancedLightGCN(LightGCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加新组件
```

### 3. 多行为建模

支持多种用户行为的联合建模：

```python
dataset = AlibabaDataset(
    behavior_types=[1, 2, 3, 4],  # 浏览、收藏、加购、购买
    multi_behavior=True
)
```

## 📈 可视化功能

### 1. 训练过程可视化

```python
trainer.plot_training_curves('./results/training_curves.png')
```

### 2. 嵌入空间可视化

```python
# PCA降维可视化用户和商品嵌入
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
user_emb_2d = pca.fit_transform(user_embeddings)
```

### 3. 推荐结果分析

```python
system = AlibabaRecommendationSystem(data_path, device=device)
system.analyze_recommendations(user_id=0, k=20)
```

## ⚡ 性能优化

### 1. GPU加速

```bash
# 自动检测并使用GPU
python main_alibaba.py --mode train  # 自动使用CUDA
```

### 2. 批量处理

```python
# 增大批次大小以提高训练效率
trainer = LightGCNTrainer(batch_size=2048)
```

### 3. 模型并行

对于大规模数据，可以使用数据并行：

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 🐛 常见问题

### Q1: 内存不足怎么办？
- 减小 `batch_size` 和 `embedding_dim`
- 使用梯度累积
- 启用混合精度训练

### Q2: 训练速度慢？
- 增大 `batch_size`
- 使用GPU加速
- 减少 `num_layers` 或 `embedding_dim`

### Q3: 推荐效果不好？
- 增加训练轮数
- 调整学习率
- 尝试不同的 `num_layers`
- 检查数据质量

## 📚 参考资料

### 学术论文
- **LightGCN**: [Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)
- **BPR**: [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)

### 数据集
- [阿里移动推荐算法挑战赛](https://tianchi.aliyun.com/competition/entrance/1/information)
- [长期学习赛](https://tianchi.aliyun.com/competition/entrance/532043/information)

### 相关资源
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [推荐系统实践](https://github.com/microsoft/recommenders)

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置

```bash
git clone <repository>
cd lightgcn-alibaba
pip install -r requirements_alibaba.txt
```

### 代码规范
- 使用中文注释说明核心逻辑
- 遵循PEP 8代码风格
- 添加类型提示
- 编写测试用例

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

- 感谢阿里巴巴提供的移动推荐数据集
- 感谢LightGCN论文作者的开源贡献
- 感谢PyTorch Geometric团队的优秀工作

---

如果本项目对您有帮助，请给个⭐️支持一下！

如有问题或建议，欢迎提Issue或联系维护者。