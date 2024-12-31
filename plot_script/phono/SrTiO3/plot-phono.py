import yaml
import numpy as np
import matplotlib.pyplot as plt

def plot_phonon_band_structure(band_yaml_path):
    """
    读取 phonopy 生成的 band.yaml 文件并绘制声子谱。
    
    参数:
    - band_yaml_path: str, band.yaml 文件的路径
    """
    # 读取 band.yaml 文件
    with open(band_yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # 提取基本信息
    nqpoint = data.get('nqpoint', 0)
    npath = data.get('npath', 0)
    segment_nqpoint = data.get('segment_nqpoint', [])
    labels_segments = data.get('labels', [])
    frequency_conversion = data.get('frequency_unit_conversion_factor', 1.0)
    
    # 提取 q点信息
    phonon_data = data.get('phonon', [])
    
    # 初始化列表以存储距离和频率
    distances = []
    frequencies = []
    
    # 假设所有 q点的声子数量相同，获取声子数量
    if len(phonon_data) > 0:
        nbands = len(phonon_data[0]['band'])
    else:
        raise ValueError("phonon 数据为空，请检查 band.yaml 文件。")
    
    # 初始化一个 nbands x nqpoint 的数组来存储频率
    frequencies = np.zeros((nqpoint, nbands))
    
    # 填充 distances 和 frequencies
    for i, qpoint in enumerate(phonon_data):
        distances.append(qpoint['distance'])
        for band_idx, band in enumerate(qpoint['band']):
            frequencies[i, band_idx] = band['frequency'] * frequency_conversion  # 转换单位
    
    distances = np.array(distances)
    
    # 计算高对称点的位置
    high_symmetry_distances = []
    high_symmetry_labels = []
    current = 0
    high_symmetry_distances.append(distances[0])
    high_symmetry_labels.append(labels_segments[0][0])  # 第一个高对称点标签
    
    for idx, nq in enumerate(segment_nqpoint):
        if idx < len(labels_segments):
            # 每个段的结束点对应下一个段的起始高对称点
            current += nq
            if current < len(distances):
                high_symmetry_distances.append(distances[current -1])
                high_symmetry_labels.append(labels_segments[idx][1])
    
    # 确保最后一个高对称点被添加
    if high_symmetry_distances[-1] != distances[-1]:
        high_symmetry_distances.append(distances[-1])
        high_symmetry_labels.append(labels_segments[-1][1])
    
    # 绘制声子谱
    plt.figure(figsize=(10, 6))
    
    for band in range(nbands):
        plt.plot(distances, frequencies[:, band], color='b', linewidth=0.5)
    
    # 添加高对称点的垂直线和标签
    for d, label in zip(high_symmetry_distances, high_symmetry_labels):
        plt.axvline(x=d, color='k', linestyle='--', linewidth=0.5)
        plt.text(d, plt.ylim()[0] - (plt.ylim()[1]-plt.ylim()[0])*0.05, label, 
                 horizontalalignment='center', verticalalignment='top', fontsize=12)
    
    # 设置 x 轴刻度
    plt.xticks(high_symmetry_distances, high_symmetry_labels, fontsize=12)
    
    # 设置标签和标题
    plt.xlabel('Wave Vector Path', fontsize=14)
    plt.ylabel('Frequency (THz)', fontsize=14)
    plt.title('Phonon Band Structure', fontsize=16)
    
    # 设置 y 轴的范围，考虑负频率
    y_min = min(np.min(frequencies), 0)
    y_max = max(np.max(frequencies), 0)
    plt.ylim(y_min - (y_max - y_min)*0.05, y_max + (y_max - y_min)*0.05)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 指定 band.yaml 文件的路径
    band_yaml_path = 'band-lmp.yaml'  # 如果文件在其他路径，请修改此行
    
    # 绘制声子谱
    plot_phonon_band_structure(band_yaml_path)
