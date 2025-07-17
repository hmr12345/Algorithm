import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    # 创建一个二维数组，行数为时间步数，列数为120个触觉点。
    # bug:Why only z?
    tactile_z = np.zeros((len(df), 120))  # 时间步×120点
    for i in range(120):
        col = f'1-2-12x10-Z[{i}]'
        tactile_z[:, i] = df[col].values
    
    # 关键点：将触觉矩阵展平为时序特征
    tactile_features = tactile_z.reshape(len(df), -1)  # [时间步, 120]
    features['tactile_seq'] = tactile_features
    
    return features

# 测试特征提取
features = extract_features(sample_df)
print(f"全局特征维度: {len(features) - 1}")  # 减去'tactile_seq'
print(f"触觉序列形状: {features['tactile_seq'].shape}")