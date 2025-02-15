import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体和样式
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def read_and_process_data(filepath):
    """读取数据并按温度计算平均值"""
    # 读取数据，只取温度和晶格常数列
    df = pd.read_csv(filepath,
                     sep='\s+',
                     usecols=[1, 7, 8, 9],
                     names=['Temp', 'Lx', 'Ly', 'Lz'])

    # 将温度四舍五入到最近的25的倍数
    df['Temp_rounded'] = (df['Temp'] / 25).round() * 25

    # 计算每个温度点的平均值
    means = df.groupby('Temp_rounded').mean()

    # 转换单位并处理Lz
    temp_points = means.index.values
    lx_values = means['Lx'].values / 10
    ly_values = means['Ly'].values / 10
    lz_values = means['Lz'].values / 10

    return temp_points, lx_values, ly_values, lz_values


def plot_lattice_constants(temp, lx, ly, lz, save_path=None):
    """绘制晶格常数随温度的变化曲线"""
    plt.figure(figsize=(10, 7))
    
    # Nature配色方案
    colors = {
        'a': '#E64B35',  # 红色
        'b': '#4DBBD5',  # 蓝色
        'c': '#00A087'   # 绿色
    }
    
    # 设置绘图样式
    plt.plot(temp, lx, 'o-', color=colors['a'],
             label='a', linewidth=2, markersize=8)
    plt.plot(temp, ly, 's-', color=colors['b'],
             label='b', linewidth=2, markersize=8)
    plt.plot(temp, lz, '^-', color=colors['c'],
             label='c', linewidth=2, markersize=8)

    # 设置图表样式
    plt.xlabel('T(K)', fontsize=14)
    plt.ylabel('Lattice constant ($\mathrm{\AA}$)', fontsize=14)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.3)

    # 设置刻度样式
    plt.tick_params(direction='in', length=6, width=1)
    plt.tick_params(which='minor', direction='in', length=3, width=1)
    plt.minorticks_on()

    # 设置坐标轴范围和刻度
    plt.xlim(-10, 310)
    plt.ylim(3.935, 3.97)
    plt.xticks(np.arange(0, 301, 50), fontsize=12)
    plt.yticks(fontsize=12)

    # 设置背景色为浅灰色
    # plt.gca().set_facecolor('#F6F6F6')
    # plt.grid(True, linestyle='--', alpha=0.5, color='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    # 文件路径
    data_file = 'i4mcm/lmps-1.dat'

    # 读取和处理数据
    temperatures, lx_data, ly_data, lz_data = read_and_process_data(data_file)

    # 绘制图表
    plot_lattice_constants(temperatures, lx_data, ly_data,
                           lz_data, 'lattice_constants.png')
