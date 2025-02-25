import numpy as np
import pandas as pd
import os
from data_utils import data_processor


from caesar.logger.logger import setup_logger

logger = setup_logger(__name__)

class PolarizationCalculator:
    def __init__(self):
        # Born有效电荷
        self.Z_Pb = 3.9  # Pb的Born有效电荷
        self.Z_Sr = 2.6  # Sr的Born有效电荷
        self.Z_Ti = 7.1  # Ti的Born有效电荷
        self.Z_O = -2.1  # O的Born有效电荷
        
        # 单位晶胞体积 (Å³)
        self.V_uc = 62.5  # 假设值，需要根据实际情况调整
        
        # Sr浓度 (1-α 为Sr浓度)
        self.alpha = 0.5  # PbxSr(1-x)TiO3中的x值
        
        # 计算平均Born有效电荷
        self.Z_A = self.alpha * self.Z_Pb + (1 - self.alpha) * self.Z_Sr

    def calculate_unit_cell_polarization(self, atoms_coords):
        """
        计算单个单位晶胞的极化
        
        参数:
        atoms_coords: 包含原子类型和坐标的列表
        每个元素格式: [atom_type, x, y, z]
        
        返回:
        numpy数组: [Px, Py, Pz]
        """
        P = np.zeros(3)
        A_site_coords = []
        Ti_coords = []
        O_coords = []
        
        # 分类原子坐标
        for atom in atoms_coords:
            atom_type = atom[0]
            coords = np.array(atom[1:4])
            
            if atom_type == 1:  # A-site (Pb/Sr)
                A_site_coords.append(coords)
            elif atom_type == 2:  # Ti
                Ti_coords.append(coords)
            elif atom_type == 3:  # O
                O_coords.append(coords)
        
        # 计算极化
        if len(A_site_coords) == 8 and len(Ti_coords) == 1 and len(O_coords) == 6:
            # A-site贡献
            P += (self.Z_A / 8) * np.sum(A_site_coords, axis=0)
            # Ti贡献
            P += self.Z_Ti * Ti_coords[0]
            # O贡献
            P += (self.Z_O / 2) * np.sum(O_coords, axis=0)
            
            # 除以单位晶胞体积
            P = P / self.V_uc
            
        return P

    def read_trajectory_file(self, filename):
        """
        读取轨迹文件
        待实现
        """
        pass




