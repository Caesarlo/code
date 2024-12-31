import numpy as np
import yaml

from pathlib import Path
from typing import List


class Phonon:
    def __init__(self, filename: str):
        """
        初始化 Phonon 类并验证文件路径和格式。

        :param filename: YAML 文件路径
        """
        if not isinstance(filename, str) or not filename:
            raise ValueError("文件路径必须是非空字符串")

        self.filename = Path(filename)

        if not self.filename.exists():
            raise FileNotFoundError(f"文件不存在: {self.filename}")

        if self.filename.suffix.lower() != '.yaml':
            raise ValueError("文件格式错误，仅支持 .yaml 文件")

        self.data = {}
        self.attributes = {}
        self.load_file()

    def load_file(self):
        try:
            with open(self.filename, 'r') as f:
                self.data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 文件解析错误: {e}")
        except Exception as e:
            raise IOError(f"文件读取错误: {e}")

    def get_attributes(self):
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
        else:
            raise ValueError("请先加载文件")


p = Phonon('band.yaml')
