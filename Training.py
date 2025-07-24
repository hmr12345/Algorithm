import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from Models.transformer import TransformerClassifier
import torch.nn as nn

# 数据转为PyTorch张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)
dataset = TensorDataset(X_tensor, y_tensor)

# 划分训练集/验证集
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# 初始化模型
model = TransformerClassifier(input_dim=120, d_model=128, num_heads=8, num_layers=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(50):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_acc = sum((model(batch_x).argmax(1) == batch_y).sum().item() 
                      for batch_x, batch_y in val_loader) / len(val_data)
    print(f"Epoch {epoch}: Val Acc={val_acc:.3f}")