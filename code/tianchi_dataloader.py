"""
天池移动电商推荐数据集加载器
"""
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
from time import time
import world
from world import cprint
from dataloader import BasicDataset


class TianchiDataset(BasicDataset):
    """
    天池移动电商推荐数据集
    """
    def __init__(self, config=world.config, path="../data/tianchi"):
        cprint(f'loading tianchi dataset from [{path}]')
        
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.path = path
        
        # 读取用户行为数据
        user_file = os.path.join(path, 'tianchi_mobile_recommend_train_user.csv')
        item_file = os.path.join(path, 'tianchi_mobile_recommend_train_item.csv')
        
        self.user_df = pd.read_csv(user_file)
        self.item_df = pd.read_csv(item_file)
        
        # 数据预处理
        self._preprocess_data()
        
        # 构建训练和测试数据
        self._build_train_test_data()
        
        # 构建用户-物品交互矩阵
        self._build_user_item_matrix()
        
        # 构建图
        self.Graph = None
        
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"tianchi Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
        print("tianchi dataset is ready to go")

    def _preprocess_data(self):
        """数据预处理"""
        # 解析时间，从时间戳中提取日期（如2014-12-18）
        self.user_df['date_str'] = self.user_df['time'].astype(str).str.slice(0, 8)  # 取前8位：20141218  
        self.user_df['date'] = pd.to_datetime(self.user_df['date_str'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # 只考虑购买行为(behavior_type=4)作为正样本
        # 但在构建图时可以使用所有行为类型
        purchase_df = self.user_df[self.user_df['behavior_type'] == 4].copy()
        
        # 筛选商品子集P中的商品
        item_subset = set(self.item_df['item_id'].values)
        purchase_df = purchase_df[purchase_df['item_id'].isin(item_subset)]
        
        # 重新编码用户和物品ID，从0开始
        unique_users = sorted(purchase_df['user_id'].unique())
        unique_items = sorted(purchase_df['item_id'].unique())
        
        user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
        
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.reverse_user_id_map = {v: k for k, v in user_id_map.items()}
        self.reverse_item_id_map = {v: k for k, v in item_id_map.items()}
        
        # 映射ID
        purchase_df['user_id_new'] = purchase_df['user_id'].map(user_id_map)
        purchase_df['item_id_new'] = purchase_df['item_id'].map(item_id_map)
        
        # 删除无法映射的行
        purchase_df = purchase_df.dropna(subset=['user_id_new', 'item_id_new'])
        purchase_df['user_id_new'] = purchase_df['user_id_new'].astype(int)
        purchase_df['item_id_new'] = purchase_df['item_id_new'].astype(int)
        
        self.purchase_df = purchase_df
        self.n_user = len(unique_users)
        self.m_item = len(unique_items)

    def _build_train_test_data(self):
        """构建训练和测试数据"""
        # 按时间划分：2014-12-17之前为训练，2014-12-18为测试
        train_df = self.purchase_df[self.purchase_df['date'] < '2014-12-18']
        test_df = self.purchase_df[self.purchase_df['date'] == '2014-12-18']
        
        # 训练数据
        self.trainUser = train_df['user_id_new'].values
        self.trainItem = train_df['item_id_new'].values
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.traindataSize = len(self.trainUser)
        
        # 测试数据
        self.testUser = test_df['user_id_new'].values
        self.testItem = test_df['item_id_new'].values
        self.testUniqueUsers = np.unique(self.testUser)
        self.testDataSize = len(self.testUser)
        
        # 构建测试字典
        self.__testDict = self.__build_test()

    def _build_user_item_matrix(self):
        """构建用户-物品交互矩阵"""
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        
        # 构建allPos
        self._allPos = self.getUserPosItems(list(range(self.n_user)))

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        """分割邻接矩阵"""
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        """将scipy稀疏矩阵转换为pytorch稀疏张量"""
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        """获取稀疏图"""
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """构建测试字典"""
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """获取用户-物品反馈"""
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        """获取用户正样本物品"""
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        return user

    def __len__(self):
        return len(self.trainUniqueUsers)