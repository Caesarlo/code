import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy.typing as npt
import seaborn as sns
from dataclasses import dataclass

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from matplotlib import gridspec
from caesar.logger.logger import setup_logger
from pathlib import Path
from typing import List, Dict, Any, Tuple
from matplotlib import rcParams


config = {
    "font.family": 'serif',
    # "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


logger = setup_logger(__name__)

def read_coordinates(file_path):
    """
    读取坐标文件并返回numpy数组
    支持两种格式：
    1. [x y z] 空格分隔
    2. [x, y, z] 逗号分隔
    
    参数:
        file_path: 文件路径字符串
    返回:
        coordinates: 包含所有坐标的numpy数组
    """
    try:
        coordinates = []
        with open(file_path, 'r') as file:
            for line in file:
                # 去除方括号、空白字符和逗号
                line = line.strip().replace('[', '').replace(']', '').replace(',', ' ')
                if line:  # 确保不是空行
                    # 分割字符串并过滤掉空字符串
                    coords = [float(x) for x in line.split() if x]
                    coordinates.append(coords)
        
        # 转换为numpy数组
        coordinates = np.array(coordinates)
        
        logger.info(f"成功读取了 {len(coordinates)} 个坐标点")
        logger.info(f"数组形状: {coordinates.shape}")
        
        return coordinates
        
    except Exception as e:
        logger.exception(f"读取文件时发生错误: {str(e)}")
        return None

Ti_i4mcm_coord_y0_list=read_coordinates('./data/Ti_i4mcm_coord_y0_list')
Ti_pm3m_coord_y0_list=read_coordinates('./data/Ti_pm3m_coord_y0_list')


# 计算每个位置的偏移向量
def calculate_offset_vectors(i4mcm_coord_y0_list,pm3m_coord_y0_list):
    offset_vectors = []
    for i in range(len(i4mcm_coord_y0_list)):
        offset_vector = i4mcm_coord_y0_list[i] - pm3m_coord_y0_list[i]
        offset_vectors.append(offset_vector)
    return offset_vectors

Ti_offset_vectors=np.array(calculate_offset_vectors(Ti_i4mcm_coord_y0_list,Ti_pm3m_coord_y0_list))


# 计算向量与x轴的夹角

def calculate_angles(vectors):
    """
    计算向量与x轴的夹角（以度为单位）
    """
    angles = []
    for vector in vectors:
        # 使用arctan2计算角度，将弧度转换为度
        angle = np.degrees(np.arctan2(vector[2], vector[0]))  # 使用z和x分量
        angles.append(angle)
    return np.array(angles)

angle_deg=calculate_angles(Ti_offset_vectors)

df=pd.DataFrame(Ti_pm3m_coord_y0_list[:,[0,2]],columns=['x','z'])
df['angle']=angle_deg
data=df.pivot(index='z',columns='x',values='angle')


plt.figure(figsize=(1,10))
ax = sns.heatmap(data, fmt='d', linewidths=.5,cmap='YlGnBu')
# 获取网格中心点的坐标
x_centers = np.arange(len(data.columns)) + 0.5
z_centers = np.arange(len(data.index)) + 0.5
X, Z = np.meshgrid(x_centers, z_centers)

# 准备箭头的 U（x方向）和 V（z方向）分量
# 将偏移向量归一化以保持箭头长度一致
normalized_vectors = Ti_offset_vectors / np.linalg.norm(Ti_offset_vectors, axis=1)[:, np.newaxis]
U = normalized_vectors[:, 0].reshape(X.shape) * 0.3  # 缩放因子0.3可以调整箭头大小
V = normalized_vectors[:, 2].reshape(X.shape) * 0.3

# 在热力图上绘制箭头
ax.quiver(X, Z, U, V, angles='xy', scale_units='xy', scale=1, color='black', width=0.015)

plt.show()