import numpy as np
import yaml
import json
import matplotlib.pyplot as plt

from matplotlib import gridspec
from caesar.logger.logger import setup_logger
from pathlib import Path
from typing import List, Dict, Any
from scipy.stats import gaussian_kde
from typing import List, Tuple
import numpy.typing as npt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    # "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


logger = setup_logger(__name__)


class Phonon:
    def __init__(self, filename: str):
        """
        初始化 Phonon 类并验证文件路径和格式。

        :param filename: YAML 文件路径
        """
        if not isinstance(filename, str) or not filename:
            logger.error("文件路径必须是非空字符串")
            raise ValueError("文件路径必须是非空字符串")

        self.filename = Path(filename)

        if not self.filename.exists():
            logger.error(f"文件不存在: {self.filename}")
            raise FileNotFoundError(f"文件不存在: {self.filename}")

        if self.filename.suffix.lower() != '.yaml':
            logger.error("文件格式错误，仅支持 .yaml 文件")
            raise ValueError("文件格式错误，仅支持 .yaml 文件")

        self.data = {}
        self.attributes = {}
        self._parse_file()
        self._parse_content()

    def __repr__(self):
        return f"Phonon(nqpoint={self.data.get('nqpoint')}, npath={self.data.get('npath')})"

    def _parse_file(self) -> Dict[str, Any]:
        """
        转换 band.yaml 文件为 Python 字典。

        :param param: None
        :type param: None
        :return: data
        :rtype: dict
        """
        try:
            with open(self.filename, 'r') as f:
                self.data = yaml.safe_load(f)
                logger.info("文件读取成功")
        except yaml.YAMLError as e:
            logger.error(f"YAML 文件解析错误: {e}")
            raise ValueError(f"YAML 文件解析错误: {e}")
        except Exception as e:
            logger.error(f"文件读取错误: {e}")
            raise IOError(f"文件读取错误: {e}")

    def _parse_content(self) -> Dict[str, Any]:
        """
        解析 band.yaml 文件内容。

        :param param: None
        :type param: None
        :return: attributes
        :rtype: Dict[str, Any]
        """
        if self.data is not None:
            self.attributes['nqpoints'] = self.data['nqpoint']  # q点数量
            self.attributes['npath'] = self.data['npath']  # 路径数量
            # 每个路径的q点数量
            self.attributes['segment_nqpoint'] = self.data['segment_nqpoint']
            self.attributes['labels'] = self.data['labels']  # 路径标签
            # 倒格矢
            self.attributes['reciprocal_lattice'] = self.data['reciprocal_lattice']
            self.attributes['natom'] = self.data['natom']  # 原子数量
            self.attributes['lattice'] = self.data['lattice']  # 晶格矢量
            self.attributes['points'] = self.data['points']  # q点坐标
            self.attributes['frequency_conversion'] = self.data.get(
                'frequency_unit_conversion_factor', 1.0)  # 频率单位转换因子
            logger.info("文件解析成功")
        else:
            logger.error("请先加载文件")
            raise ValueError("请先加载文件")

    def get_attribute(self) -> Dict:
        """
        解析 band.yaml 文件内容并返回。

        :param param: None
        :type param: None
        :return: attributes
        :rtype: Dict[str, Any]
        """
        if self.attributes is not None:
            logger.info("获取文件属性成功")
            return self.attributes
        else:
            logger.error("请先加载文件")
            raise ValueError("请先加载文件")

    def get_q_position_data(self, q_position: List[float]) -> Dict[str, Any]:
        """
        查询特定 q-position 的数据。

        :param q_position: 查询的 q-position 值
        :type q_position: List[float]
        :return: 对应 phonon 数据
        :rtype: dict
        """
        for phonon in self.data.get('phonon', []):
            if phonon['q-position'] == q_position:
                logger.info('找到对应 q-position 的数据')
                return phonon
            logger.error('未找到对应 q-position 的数据')
        return {}

    def get_atom_info(self, symbol: str) -> List[Dict[str, Any]]:
        """
        将解析后的数据导出为 JSON格式。

        :param symbol: 元素符号
        :type symbol: str
        :return: 包含所有匹配原子的信息列表
        :rtype: List[Dict[str, Any]]
        """
        return [point for point in self.data.get('points', []) if point['symbol'] == symbol]

    def export_to_json(self, output_path: str):
        """
        将解析后的数据导出为 JSON 格式

        :param output_path: 输出文件路径
        :type output_path: str
        """
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def visualize_frequencies(self):
        """
        可视化声子谱和DOS。
        """
        logger.info("绘制声子谱和DOS")
        plot_phonon_band_structure_and_dos_single(
            self.filename, label='Example', color='blue', dos_sigma=0.05)





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
    ax_dos.set_yticklabels([]) # 移除默认刻度标签

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

    ax_dos.set_xlabel('DOS (arb. units)', fontsize=14, labelpad=10)
    ax_dos.set_title('Phonon DOS', fontsize=16)
    ax_dos.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    band_yaml_path = 'band-lmp.yaml'
    plot_phonon_band_structure_and_dos_single(
        band_yaml_path, label='Example', color='blue', dos_sigma=0.05)
