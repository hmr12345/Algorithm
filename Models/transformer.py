import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码（支持1D时序或2D空间位置）"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # 线性变换并分头 [batch, seq_len, num_heads, d_k]
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        # 加权求和
        output = torch.matmul(attn_weights, v)
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)

class TransformerBlock(nn.Module):
    """单个Transformer层（多头注意力 + 前馈网络）"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差连接 + LayerNorm
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class TransformerClassifier(nn.Module):
    """完整Transformer分类模型"""
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_classes, max_len=100):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # 分类token
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_dim] (如触觉序列或图像分块)
        x = self.embedding(x)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)
        # 添加CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 通过Transformer层
        for block in self.transformer_blocks:
            x = block(x)
        # 取CLS token输出分类
        cls_output = x[:, 0]
        return self.classifier(cls_output)


#--------------------------------#
#       训练模型（示例）          #
#--------------------------------#
# 参数配置
input_dim = 64   # 输入特征维度（如触觉信号展平后的长度）
d_model = 128    # Transformer隐藏层维度
num_heads = 8    # 注意力头数
num_layers = 4   # Transformer层数
num_classes = 3  # 分类类别数（如安全/滑动/损坏）

# 初始化模型
model = TransformerClassifier(input_dim, d_model, num_heads, num_layers, num_classes)

# 模拟输入数据 (batch_size=2, seq_len=10)
x = torch.randn(2, 10, input_dim)  # 假设10个时间步的触觉序列

# 前向传播
output = model(x)  # 输出形状: [2, num_classes]
print(output)
