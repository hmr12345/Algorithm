import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

#--------------------------------
#       加载数据
#--------------------------------
def load_data(class_label, sample_id):
    """加载单个CSV文件"""
    # 需要修改！
    path = f"./Dataset/class_{class_label}/sample_{sample_id}.csv"
    df = pd.read_csv(path)
    # 去掉第一列（时间戳），只保留合力和120个点的xyz力
    df = df.iloc[:, 1:]
    return df

# 示例：加载类别0的第一个样本
sample_df = load_data(class_label=0, sample_id=1)
print(sample_df.head())

#--------------------------------
#       特征工程
#--------------------------------
def extract_features(df):# 一个pandas的DataFrame，包含120个触觉点的xyz力
    """
    提取关键特征：
    - 全局特征：合力X/Y/Z的统计量
    - 局部特征：120个触觉点的时序特征
    """
    features = {}
    
    # 1. 全局合力特征（均值/方差/最大值）
    for axis in ['X', 'Y', 'Z']:
        col = f'1-2-1x1-{axis}'
        features[f'global_{axis}_mean'] = df[col].mean() #计算该方向合力的均值，存入字典
        features[f'global_{axis}_std'] = df[col].std() #计算该方向合力的标准差，存入字典
    
    # 2. 触觉点特征（每个点的Z轴均值+120点矩阵的SVD分解）
    # 创建三个二维数组，分别存储X、Y、Z方向的触觉点数据
    # Question: xyz三个方向的feature是否比单个z方向的feature更好？
    tactile_x = np.zeros((len(df), 120))
    tactile_y = np.zeros((len(df), 120))
    tactile_z = np.zeros((len(df), 120))

    for i in range(120):
        col_x = f'1-2-12x10-X[{i}]'
        col_y = f'1-2-12x10-Y[{i}]'
        col_z = f'1-2-12x10-Z[{i}]'
        tactile_x[:, i] = df[col_x].values
        tactile_y[:, i] = df[col_y].values
        tactile_z[:, i] = df[col_z].values

    # 如果你想把三轴数据拼接成一个大特征矩阵（shape: [样本数, 120*3]）：
    tactile_xyz = np.concatenate([tactile_x, tactile_y, tactile_z], axis=1)
    
    # 关键点：将触觉矩阵展平为时序特征
    tactile_features = tactile_xyz.reshape(len(df), -1)  # [时间步, 120]
    features['tactile_seq'] = tactile_features
    
    return features

# 测试特征提取
features = extract_features(sample_df)
print(f"全局特征维度: {len(features) - 1}")  # 减去'tactile_seq'
print(f"触觉序列形状: {features['tactile_seq'].shape}")

def preprocess_all_data():
    """处理所有样本"""
    all_sequences = []
    labels = []
    
    for class_label in [0, 1, 2]:
        for sample_id in range(1, 101):
            df = load_data(class_label, sample_id)
            features = extract_features(df)
            
            # 标准化触觉序列（按样本独立标准化）
            scaler = StandardScaler()
            seq_normalized = scaler.fit_transform(features['tactile_seq'])
            
            all_sequences.append(seq_normalized)
            labels.append(class_label)
    
    # 统一填充到最大长度（保持时间步维度）
    max_len = max(len(seq) for seq in all_sequences)
    X = pad_sequences(all_sequences, maxlen=max_len, dtype='float32', padding='post')
    y = np.array(labels)
    
    return X, y

X, y = preprocess_all_data()
print(f"最终数据形状: {X.shape}")  # (300样本, 最大时间步, 120特征)