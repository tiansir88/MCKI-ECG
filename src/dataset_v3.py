import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import numpy.core as _nc
sys.modules["numpy._core"] = _nc
sys.modules["numpy._core.multiarray"] = _nc.multiarray


# === 🏷️ 映射与优先级配置 ===

# 临床严重程度等级 (用于在多标签中选出一个“主标签”作为对比学习的 Anchor)
# 逻辑：MI (梗死) > HYP (肥大) > CD (传导) > STTC (ST-T改变) > NORM (正常)
SEVERITY_RANK = {
    'MI': 5,
    'HYP': 4,
    'CD': 3,
    'STTC': 2,
    'NORM': 1
}

CLASS_MAP = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}


class PTBXLDatasetV3(Dataset):
    """
    针对多标签优化的 Dataset
    返回值:
        - signal: (12, 1000) 信号张量
        - multi_hot: (5,) 二进制标签向量 (用于 BCE 训练)
        - anchor_label: 整数索引 (用于 KG-Guided 对比学习)
    """

    def __init__(self, X_path, y_raw_path, y_mh_path, transform=None):
        # 加载信号数据 (N, 1000, 12)
        self.X = np.load(X_path)
        # 加载原始标签列表 (用于判断优先级), e.g., [['MI', 'STTC'], ['NORM'], ...]
        self.y_raw = np.load(y_raw_path, allow_pickle=True)
        # 加载 Multi-hot 矩阵 (N, 5)
        self.y_mh = np.load(y_mh_path)

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def get_primary_label_index(self, labels_list):
        """
        核心逻辑：从多个标签中挑出一个最严重的作为对比学习的中心点（Anchor）
        """
        if len(labels_list) == 0:
            return 0  # 默认 NORM

        # 按严重程度排序，选最高分的
        best_label = 'NORM'
        max_severity = -1

        for l in labels_list:
            severity = SEVERITY_RANK.get(l, 0)
            if severity > max_severity:
                max_severity = severity
                best_label = l

        return CLASS_MAP.get(best_label, 0)

    def __getitem__(self, idx):
        # 1. 处理信号
        # 将形状从 (1000, 12) 转置为 (12, 1000)，适配 ResNet1d 的输入 [Channel, Length]
        signal = self.X[idx].transpose()
        signal = torch.tensor(signal, dtype=torch.float32)

        if self.transform:
            signal = self.transform(signal)

        # 2. 获取 Multi-hot 标签 (用于分类训练)
        multi_hot = torch.tensor(self.y_mh[idx], dtype=torch.float32)

        # 3. 获取主标签索引 (用于对比学习)
        raw_labels = self.y_raw[idx]
        anchor_label = self.get_primary_label_index(raw_labels)

        return signal, multi_hot, anchor_label


# === 🚚 DataLoader 工厂函数 ===

def get_dataloader_v3(data_dir, batch_size=64, num_workers=4):
    """
    根据 data_dir 自动加载训练、验证和测试集
    """
    # 定义文件路径 (对应 preprocess_v3.py 的输出)
    train_files = {
        'X': os.path.join(data_dir, 'X_train.npy'),
        'raw': os.path.join(data_dir, 'y_train.npy'),
        'mh': os.path.join(data_dir, 'y_train_mh.npy')
    }
    val_files = {
        'X': os.path.join(data_dir, 'X_val.npy'),
        'raw': os.path.join(data_dir, 'y_val_mh.npy'),  # 验证集通常只需 mh
        'mh': os.path.join(data_dir, 'y_val_mh.npy')
    }
    test_files = {
        'X': os.path.join(data_dir, 'X_test.npy'),
        'raw': os.path.join(data_dir, 'y_test_raw.npy'),
        'mh': os.path.join(data_dir, 'y_test_mh.npy')
    }

    # 创建 Dataset 实例
    train_ds = PTBXLDatasetV3(train_files['X'], train_files['raw'], train_files['mh'])
    # 注意：验证集和测试集如果不需要对比学习，raw 参数可以传 mh
    val_ds = PTBXLDatasetV3(val_files['X'], val_files['mh'], val_files['mh'])
    test_ds = PTBXLDatasetV3(test_files['X'], test_files['raw'], test_files['mh'])

    # 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
