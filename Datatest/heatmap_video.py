import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib import cm
import os
from PIL import Image
import imageio

def process_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 处理时间戳相同的行 - 取平均值
    df = df.groupby('Timestamp').mean().reset_index()
    
    # 提取时间戳
    timestamps = df['Timestamp']
    
    # 提取测点数据
    # 每个测点有X,Y,Z三个分力，共120个测点
    # 测点命名格式为1-2-12x10-X[0],1-2-12x10-Y[0],1-2-12x10-Z[0],...,1-2-12x10-Z[119]
    # 我们需要提取所有测点的X,Y,Z分力
    
    # 创建测点数据数组
    # 形状为(时间点数量, 120, 3) - 3代表X,Y,Z
    num_timepoints = len(df)
    num_sensors = 120
    sensor_data = np.zeros((num_timepoints, num_sensors, 3))
    
    for i in range(num_sensors):
        x_col = f'1-2-12x10-X[{i}]'
        y_col = f'1-2-12x10-Y[{i}]'
        z_col = f'1-2-12x10-Z[{i}]'
        
        sensor_data[:, i, 0] = df[x_col].values
        sensor_data[:, i, 1] = df[y_col].values
        sensor_data[:, i, 2] = df[z_col].values
    
    return timestamps, sensor_data

def create_sensor_grid(sensor_data, time_idx, component='magnitude'):
    """
    创建传感器数据的网格
    component: 'x', 'y', 'z' 或 'magnitude'(合力)
    """
    # 假设传感器排列是12x10的网格
    rows = 12
    cols = 10
    
    if component == 'magnitude':
        # 计算合力大小
        grid_data = np.sqrt(np.sum(sensor_data[time_idx]**2, axis=1))
    elif component == 'x':
        grid_data = sensor_data[time_idx, :, 0]
    elif component == 'y':
        grid_data = sensor_data[time_idx, :, 1]
    elif component == 'z':
        grid_data = sensor_data[time_idx, :, 2]
    else:
        raise ValueError("component must be 'x', 'y', 'z' or 'magnitude'")
    
    # 重塑为12x10网格
    grid = grid_data.reshape(rows, cols)
    
    return grid

def create_heatmap_frames(timestamps, sensor_data, output_dir='frames', component='magnitude'):
    """
    为每个时间点创建热图并保存为图片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 确定颜色范围
    all_data = np.sqrt(np.sum(sensor_data**2, axis=2)) if component == 'magnitude' else sensor_data[:, :, {'x':0, 'y':1, 'z':2}[component]]
    vmin, vmax = np.min(all_data), np.max(all_data)
    
    frames = []
    
    for i, timestamp in enumerate(timestamps):
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取当前时间点的网格数据
        grid = create_sensor_grid(sensor_data, i, component)
        
        # 创建热图
        im = ax.imshow(grid, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Force Magnitude' if component == 'magnitude' else f'Force ({component.upper()})')
        
        # 添加标题
        ax.set_title(f'Sensor Force Heatmap\n{timestamp}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # 保存图像
        frame_path = os.path.join(output_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_path)
        plt.close()
        
        frames.append(frame_path)
    
    return frames

def create_video(frames, output_file='heatmap_video.mp4', fps=10):
    """
    从帧图像创建视频
    """
    # 读取所有帧
    images = [imageio.imread(frame) for frame in frames]
    
    # 保存为视频
    imageio.mimsave(output_file, images, fps=fps)

def main():
    # 输入文件路径
    input_file = r'D:\VS Repository\Algorithm\Datatest\cardboard_data1.csv'
    
    # 处理数据
    timestamps, sensor_data = process_data(input_file)
    
    # 创建热图帧 (可以选择 'x', 'y', 'z' 或 'magnitude')
    frames = create_heatmap_frames(timestamps, sensor_data, component='magnitude')
    
    # 创建视频
    create_video(frames)
    
    print("视频生成完成!")

if __name__ == "__main__":
    main()

# 横坐标（X轴）：
# 表示传感器的 列号（Column）传感器排列成一个 12行×10列 的网格（rows = 12, cols = 10），所以 X 轴的范围是 0 到 9（共 10 列）
# 纵坐标（Y轴）：
# 表示传感器的 行号（Row）
# 由于网格是 12 行 × 10 列，Y 轴的范围是 0 到 11（共 12 行）
# 颜色（Z轴）：
# 表示 力的大小（取决于 component 参数）：
# 如果 component='magnitude'，颜色表示 合力大小（sqrt(X² + Y² + Z²)）
# 如果 component='x'、'y' 或 'z'，颜色分别表示 X、Y 或 Z 方向的力