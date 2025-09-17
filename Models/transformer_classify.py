import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 数据加载和预处理
class TactileDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['cardboard', 'sponge', 'towel']  # 0, 1, 2
        self.file_paths = []
        self.labels = []
        
        # 收集所有文件路径和标签
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.csv'):
                        self.file_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(class_idx)
        
        # 初始化标准化器
        self.scaler = StandardScaler()
        self._fit_scaler()
        
    def _fit_scaler(self):
        """使用部分数据拟合标准化器"""
        sample_data = []
        for i in range(min(50, len(self.file_paths))):  # 使用前50个样本进行拟合
            df = pd.read_csv(self.file_paths[i])
            # 排除时间戳列
            features = df.drop(columns=['Timestamp']).values
            sample_data.append(features)
        
        sample_data = np.vstack(sample_data)
        self.scaler.fit(sample_data)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 加载CSV文件
        df = pd.read_csv(self.file_paths[idx])
        
        # 排除时间戳列
        features = df.drop(columns=['Timestamp']).values
        
        # 标准化特征
        features = self.scaler.transform(features)
        
        # 转换为张量
        features = torch.FloatTensor(features)
        label = torch.LongTensor([self.labels[idx]])
        
        if self.transform:
            features = self.transform(features)
            
        return features, label

# 2. 构建Transformer模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TactileTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super(TactileTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_avg_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        x = self.classifier(x)  # (batch_size, num_classes)
        return x

# 3. 训练和评估函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).squeeze()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, '
              f'Val Acc: {epoch_val_acc:.4f}')
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['cardboard', 'sponge', 'towel']))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['cardboard', 'sponge', 'towel'],
                yticklabels=['cardboard', 'sponge', 'towel'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

# 主函数
def main():
    # 参数设置
    data_dir = 'path/to/your/data'  # 替换为您的数据路径
    batch_size = 16
    d_model = 128
    nhead = 8
    num_layers = 4
    dim_feedforward = 256
    num_classes = 3
    learning_rate = 0.001
    num_epochs = 100
    
    # 加载数据集
    dataset = TactileDataset(data_dir)
    
    # 划分训练集、验证集和测试集
    train_idx, test_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=[dataset.labels[i] for i in train_idx]
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 获取输入维度（特征数量）
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[1]
    
    # 初始化模型
    model = TactileTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练模型...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )
    
    # 绘制训练过程
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 评估模型
    print("评估模型在测试集上的表现...")
    evaluate_model(model, test_loader)
    
    # 保存模型
    torch.save(model.state_dict(), 'tactile_transformer_model.pth')
    print("模型已保存为 'tactile_transformer_model.pth'")

if __name__ == '__main__':
    main()