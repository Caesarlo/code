import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import gridspec
from typing import List, Tuple
import numpy.typing as npt

plt.rc('font',family='Times New Roman')

class PhononData:
    """用于存储和处理声子数据的类"""
    def __init__(self, frequencies: npt.NDArray, distances: npt.NDArray,
                 hs_distances: List[float], hs_labels: List[str]):
        self.frequencies = frequencies
        self.distances = distances
        # 确保高对称点距离和标签数量相同
        if len(hs_distances) != len(hs_labels):
            min_len = min(len(hs_distances), len(hs_labels))
            self.hs_distances = hs_distances[:min_len]
            self.hs_labels = hs_labels[:min_len]
        else:
            self.hs_distances = hs_distances
            self.hs_labels = hs_labels
        self._cached_dos = None
        self._cached_freq_range = None

def read_band_yaml(band_yaml_path: str) -> PhononData:
    """优化的band.yaml文件读取函数"""
    with open(band_yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    phonon_data = data.get('phonon', [])
    if not phonon_data:
        raise ValueError(f"phonon数据为空: {band_yaml_path}")
    
    nqpoint = len(phonon_data)
    nbands = len(phonon_data[0]['band'])
    frequencies = np.zeros((nqpoint, nbands))
    distances = np.zeros(nqpoint)
    
    freq_conversion = data.get('frequency_unit_conversion_factor', 1.0)
    for i, qpoint in enumerate(phonon_data):
        distances[i] = qpoint['distance']
        frequencies[i] = [band['frequency'] for band in qpoint['band']]
    frequencies *= freq_conversion
    
    # 改进的高对称点处理
    segment_nqpoint = data.get('segment_nqpoint', [])
    labels_segments = data.get('labels', [])
    
    if not segment_nqpoint or not labels_segments:
        # 如果没有分段信息，使用起点和终点
        hs_distances = [distances[0], distances[-1]]
        hs_labels = ['Γ', 'X']  # 默认标签
    else:
        # 计算累积点数
        cumulative_points = np.cumsum([0] + segment_nqpoint)
        hs_distances = []
        hs_labels = []
        
        # 添加第一个点
        hs_distances.append(distances[0])
        hs_labels.append(labels_segments[0][0])
        
        # 添加中间点
        for i, idx in enumerate(cumulative_points[1:-1]):
            if idx < len(distances):
                hs_distances.append(distances[idx])
                hs_labels.append(labels_segments[i][1])
        
        # 添加最后一个点
        if cumulative_points[-1] <= len(distances):
            hs_distances.append(distances[-1])
            hs_labels.append(labels_segments[-1][1])
    
    return PhononData(frequencies, distances, hs_distances, hs_labels)

def compute_dos(phonon_data: PhononData, dos_sigma: float = 0.1) -> Tuple[npt.NDArray, npt.NDArray]:
    """优化的DOS计算函数"""
    if phonon_data._cached_dos is not None:
        return phonon_data._cached_freq_range, phonon_data._cached_dos
    
    all_frequencies = phonon_data.frequencies.ravel()
    kde = gaussian_kde(all_frequencies)
    
    freq_min, freq_max = all_frequencies.min(), all_frequencies.max()
    frequency_range = np.linspace(freq_min, freq_max, 500)
    dos = kde(frequency_range)
    
    phonon_data._cached_freq_range = frequency_range
    phonon_data._cached_dos = dos
    
    return frequency_range, dos

def plot_phonon_band_structure_and_dos_single(band_yaml_path: str, 
                                               label: str = 'Phonon Data', 
                                               color: str = 'blue', 
                                               dos_sigma: float = 0.1) -> None:
    """绘制单个声子带结构和DOS"""
    data = read_band_yaml(band_yaml_path)
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_band = plt.subplot(gs[0])
    ax_dos = plt.subplot(gs[1], sharey=ax_band)
    
    y_min, y_max = data.frequencies.min(), data.frequencies.max()
    y_padding = (y_max - y_min) * 0.05
    
    # 绘制声子带结构
    for band in range(data.frequencies.shape[1]):
        ax_band.plot(data.distances, data.frequencies[:, band], 
                     color=color, linewidth=0.5)
    
    # 确保高对称点和标签数量匹配
    for d, l in zip(data.hs_distances, data.hs_labels):
        ax_band.axvline(x=d, color='k', linestyle='--', linewidth=0.5)
        ax_band.text(d, y_min - y_padding - 1.5, l, 
                     horizontalalignment='center', 
                     verticalalignment='top', 
                     fontsize=12)
    
    # 设置刻度
    ax_band.set_xticks(data.hs_distances)
    # ax_band.set_xticklabels([])  # 移除默认刻度标签，因为我们已经用text添加了标签
    
    # 绘制DOS
    freq_range, dos = compute_dos(data, dos_sigma)
    ax_dos.plot(dos, freq_range, color=color, linewidth=0.5, label=label)
    
    # 设置图表属性
    ax_band.set_xlabel('Wave Vector Path', fontsize=14, labelpad=10)
    ax_band.set_ylabel('Frequency (THz)', fontsize=14)
    ax_band.set_title('Phonon Band Structure', fontsize=16)
    ax_band.set_ylim(y_min - y_padding, y_max + y_padding)
    ax_band.set_xlim(data.distances[0], data.distances[-1])
    ax_band.grid(True, linestyle='--', alpha=0.5)
    
    ax_dos.set_xlabel('DOS (arb. units)', fontsize=14,labelpad=10)
    ax_dos.set_title('Phonon DOS', fontsize=16)
    ax_dos.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    band_yaml_path = 'band-lmp.yaml'
    plot_phonon_band_structure_and_dos_single(band_yaml_path, label='Example', color='blue', dos_sigma=0.05)
