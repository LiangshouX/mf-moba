import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 示例数据（假设 batch、head、chunk 的范围）
batch = np.array([0, 0, 1, 1, 2, 2])  # 批次
head = np.array([0, 1, 0, 1, 0, 1])  # 头
chunk = np.array([0, 0, 1, 1, 2, 2])  # 分块
times = np.array([1.2, 1.5, 2.3, 2.8, 3.1, 3.5])  # 对应的 times 值

# 创建三维坐标系
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图，颜色映射表示 times
sc = ax.scatter(batch, head, chunk, c=times, cmap='viridis', s=50, alpha=0.8)

# 添加标签和颜色条
ax.set_xlabel('Batch')
ax.set_ylabel('Head')
ax.set_zlabel('Chunk')
plt.colorbar(sc, label='Times')

plt.title('3D Visualization of Batch, Head, Chunk vs Times')
plt.show()