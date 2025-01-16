import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import gridspec
from caesar.logger.logger import setup_logger
from pathlib import Path
from typing import List, Dict, Any
from scipy.stats import gaussian_kde
from typing import List, Tuple
import numpy.typing as npt
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    # "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def read_data(presures, space_group_symbol):
    for presure in presures:
        with open(f'{space_group_symbol}/lmps-{presure}.dat', 'r') as f:
            data_tmp = f.readlines()
        data_tmp = [i.strip().split() for i in data_tmp]
        data_df = pd.DataFrame(data_tmp, columns=[
            'Step', 'Temp', 'PotEng', 'KinEng', 'TotEng', 'Press', 'Volume', 'Lx', 'Ly', 'Lz', 'Xy', 'Xz', 'Yz'])
        temp_lattice = data_df[['Temp', 'Lx', 'Ly', 'Lz']].astype(float)

        l_x, l_y, l_z = [], [], []
        for i in range(0, 300+1, 25):
            filtered_data = temp_lattice[
                (temp_lattice['Temp'] > i-5) & (temp_lattice['Temp'] < i+5)]
            lattice_avg = list(filtered_data[['Lx', 'Ly', 'Lz']].mean())
            l_x.append(lattice_avg[0]/10)
            l_y.append(lattice_avg[1]/10)
            l_z.append(lattice_avg[2]/10)
        return [l_x, l_y, l_z]  # 返回所有方向的晶格常数

def plot_lattice_constant(data, x_rang):
    plt.figure(figsize=(12, 8))
    plt.plot(x_rang, data[0], marker='o', color='#FFCC00', linewidth=2, markersize=8, label='Lx')
    plt.plot(x_rang, data[1], marker='s', color='#0000CC', linewidth=2, markersize=8, label='Ly')
    plt.plot(x_rang, data[2], marker='^', color='#0099CC', linewidth=2, markersize=8, label='Lz')

    # 添加数据点标注
    # for i, value in enumerate(data[0]):
    #     plt.text(x_rang[i], value + 0.01, f'{value:.2f}', fontsize=10, ha='center', color='blue')
    # for i, value in enumerate(data[1]):
    #     plt.text(x_rang[i], value + 0.01, f'{value:.2f}', fontsize=10, ha='center', color='green')
    # for i, value in enumerate(data[2]):
    #     plt.text(x_rang[i], value + 0.01, f'{value:.2f}', fontsize=10, ha='center', color='red')

    # 美化图表
    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('Lattice Constants ($\mathrm{\AA}$)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    space_group_symbol = 'pm3m'
    presures = [1]
    temp_range = [i for i in range(0, 300+1, 25)]
    data = read_data(presures, space_group_symbol)
    plot_lattice_constant(data, temp_range)

