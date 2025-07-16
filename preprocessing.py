import os
import numpy as np
import pandas as pd

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

