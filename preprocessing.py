import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

'''
TODO:类别与标签映射
'''
label_map = {'sandpaper': 0, 'towel': 1, 'plastic': 2}



'''
TODO:读取数据集，并进行预处理
    对每个csv文件，去掉第一列（时间戳），只保留合力和120个点的xyz力（共363列）。
    每个csv文件的数据作为一个样本，标签为所属类别。
参数：
    root_dir：数据集根目录
返回：
    data：处理后的数据
    labels：标签
'''
def preprocess_dataset(root_dir):
    data = []
    labels = []
    for class_name, label in label_map.items():
        class_dir = os.path.join(root_dir, class_name)
        print(f"{class_dir} 文件数：", len(os.listdir(class_dir)))
        for file in os.listdir(class_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(class_dir, file)
                df = pd.read_csv(file_path)
                # 去掉时间戳，只保留合力和120个点的xyz力
                features = df.iloc[:, 1:].values  # shape: (时间步, 363)
                data.append(features)
                labels.append(label)
    return data, labels

'''
TODO:归一化数据
    对每个样本，对每个特征进行归一化，使得每个特征的均值为0，标准差为1。
参数：
    data：处理后的数据
返回：
    normed_data：归一化后的数据
    mean：每个特征的均值
    std：每个特征的标准差
'''
def normalize_data(data):
    # 假设所有样本时间步数一致
    all_data = np.concatenate(data, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    normed_data = [(d - mean) / (std + 1e-8) for d in data]
    return normed_data, mean, std

'''
TODO:主流程
'''
data_dir = 'Dataset'
data, labels = preprocess_dataset(data_dir)
# 如果需要归一化
data, mean, std = normalize_data(data)
# 转为numpy数组
labels = np.array(labels)

# # train_set = TactileDataset(X_train, y_train)
# # 假设data是list，每个元素shape为[时间步, 363]
# tensor_data = [torch.tensor(d, dtype=torch.float32) for d in data]
# # 补零，batch_first=True，输出shape: [样本数, 最大时间步, 363]
# padded_data = pad_sequence(tensor_data, batch_first=True)
# labels = torch.tensor(labels, dtype=torch.long)


# class TransformerClassifier(nn.Module):
#     def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=2, dropout=0.1):
#         super().__init__()
#         self.input_fc = nn.Linear(input_dim, model_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Linear(model_dim, num_classes)

#     def forward(self, x, mask=None):
#         # x: [batch, seq_len, input_dim]
#         x = self.input_fc(x)  # [batch, seq_len, model_dim]
#         x = self.transformer(x, src_key_padding_mask=mask)  # [batch, seq_len, model_dim]
#         # 池化：取平均
#         x = x.mean(dim=1)  # [batch, model_dim]
#         out = self.classifier(x)  # [batch, num_classes]
#         return out


# from torch.utils.data import TensorDataset, DataLoader

# # 构建数据集和dataloader
# dataset = TensorDataset(padded_data, labels)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # 实例化模型
# model = TransformerClassifier(input_dim=363, model_dim=128, num_classes=3)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()

# # 训练循环
# for epoch in range(20):
#     model.train()
#     for batch_x, batch_y in dataloader:
#         optimizer.zero_grad()
#         logits = model(batch_x)
#         loss = criterion(logits, batch_y)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
