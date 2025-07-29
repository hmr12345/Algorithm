import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取csv文件
df = pd.read_csv(r'D:\VS Repository\Algorithm\Dataset\sponge\2025-07-29-172254.csv')

# 选取所有以 '1-2-12x10-X[' 开头的列
x_cols = [col for col in df.columns if col.startswith('1-2-12x10-X[')]
data = df[x_cols]

# 将时间戳作为x轴
timestamps = df['Timestamp']

plt.figure(figsize=(15, 8))
sns.heatmap(data.T, cmap='viridis', xticklabels=20, yticklabels=10)
plt.xlabel('Time')
plt.ylabel('Sensor Index')
plt.title('Heatmap of 1-2-12x10-X[0~119]')
plt.xticks(ticks=np.arange(0, len(timestamps), max(1, len(timestamps)//10)), labels=timestamps[::max(1, len(timestamps)//10)], rotation=45)
plt.yticks(ticks=np.arange(len(x_cols)), labels=[f'X[{i}]' for i in range(len(x_cols))])
plt.tight_layout()
plt.show()

# #----分别画3张热图（xyz)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 读取csv文件
# df = pd.read_csv(r'D:\VS Repository\Algorithm\Datatest\test.csv')

# # 选取X、Y、Z列
# x_cols = [col for col in df.columns if col.startswith('1-2-12x10-X[')]
# y_cols = [col for col in df.columns if col.startswith('1-2-12x10-Y[')]
# z_cols = [col for col in df.columns if col.startswith('1-2-12x10-Z[')]

# data_x = df[x_cols]
# data_y = df[y_cols]
# data_z = df[z_cols]
# timestamps = df['Timestamp']

# fig, axes = plt.subplots(3, 1, figsize=(15, 18))
# for ax, data, label in zip(axes, [data_x, data_y, data_z], ['X', 'Y', 'Z']):
#     sns.heatmap(data.T, cmap='viridis', ax=ax, xticklabels=20, yticklabels=10)
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Sensor Index')
#     ax.set_title(f'Heatmap of 1-2-12x10-{label}[0~119]')
#     ax.set_xticks(np.arange(0, len(timestamps), max(1, len(timestamps)//10)))
#     ax.set_xticklabels(timestamps[::max(1, len(timestamps)//10)], rotation=45)
#     ax.set_yticks(np.arange(len(x_cols)))
#     # ax.set_yticklabels([f'{label}[{i}]' for i in range(len(x_cols))])
#     ax.set_yticklabels(range(len(x_cols)))# 纵坐标改为0~119
# plt.tight_layout()
# plt.show()

#----合成RGB色彩热图
# X（红色） 数据越大 → 红色越强。
# Y（绿色） 数据越大 → 绿色越强。
# Z（蓝色） 数据越大 → 蓝色越强。
# 合成颜色：
# 如果 X、Y、Z 都很大 → 接近白色（RGB=1,1,1）。
# 如果 X、Y、Z 都很小 → 接近黑色（RGB=0,0,0）。
# 如果只有 X 大 → 红色（RGB=1,0,0）。
# 如果 X 和 Y 大 → 黄色（RGB=1,1,0）。

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# df = pd.read_csv(r'D:\VS Repository\Algorithm\Datatest\test.csv')

# x_cols = [col for col in df.columns if col.startswith('1-2-12x10-X[')]
# y_cols = [col for col in df.columns if col.startswith('1-2-12x10-Y[')]
# z_cols = [col for col in df.columns if col.startswith('1-2-12x10-Z[')]

# data_x = df[x_cols].to_numpy()
# data_y = df[y_cols].to_numpy()
# data_z = df[z_cols].to_numpy()

# # 归一化到0-1
# def norm(arr):
#     arr = arr.astype(float)
#     arr = arr - np.min(arr)
#     arr = arr / (np.max(arr) + 1e-8)
#     return arr

# rgb = np.stack([norm(data_x), norm(data_y), norm(data_z)], axis=2)  # shape: (time, sensor, 3)
# rgb = np.transpose(rgb, (1, 0, 2))  # (sensor, time, 3)

# plt.figure(figsize=(15, 8))
# plt.imshow(rgb, aspect='auto')
# plt.xlabel('Time')
# plt.ylabel('Sensor Index')
# plt.title('RGB Heatmap (X=R, Y=G, Z=B)')
# plt.tight_layout()
# plt.show()