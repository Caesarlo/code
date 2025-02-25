import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy.typing as npt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


from caesar.logger.logger import setup_logger
from matplotlib import rcParams
from data_utils import LatticeDataProcessor


config = {
    "font.family": 'serif',
    # "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 减号

logger = setup_logger(__name__)


class PolarizationHeatmap:
    def __init__(self, current_data_path: str = None, ref_data_path: str = None) -> None:
        """初始化极化热力图类

        Args:
            current_data_path (str, optional): 极化相数据路径. Defaults to None.
            ref_data_path (str, optional): 立方相数据路径. Defaults to None.
        """
        self.current_data_path = current_data_path  # 极化相数据路径
        self.ref_data_path = ref_data_path  # 立方相数据路径
        self.data_processor = LatticeDataProcessor()  # 晶格数据处理器
        self.plane_number = 0  # 指定平面

        # 创建循环颜色映射
        self.create_cyclic_colormap()

    def create_cyclic_colormap(self):
        """创建循环颜色映射，用于表示向量方向

        颜色对应关系：
        - 0° (100): 红色
        - 90° (001): 绿色
        - 180° (-100): 蓝色
        - 270° (00-1): 黄色
        """
        # 定义关键方向的颜色
        colors = [
            '#FEAA95',      # 0° (100)
            '#FCFCD5',    # 90° (001)
            '#75BAFB',     # 180° (-100)
            '#BAF5A3',   # 270° (00-1)
            '#FEAA95'       # 360° (回到起点)
        ]

        # colors = ['white', 'white', 'white', 'white', 'white']
        # 创建循环颜色映射
        self.cyclic_cmap = LinearSegmentedColormap.from_list(
            'cyclic', colors, N=256)

    def read_coordinates(self, file_path: str) -> np.ndarray:
        """读取坐标文件并返回numpy数组
        支持多种格式：
        1. .npy 文件
        2. 逗号分隔的文本文件 (.csv, .txt)
        3. 空格分隔的文本文件

        Args:
            file_path (str): 数据文件路径

        Returns:
            np.ndarray: 读取的坐标数据数组
        """
        try:
            logger.info(f"读取坐标文件: {file_path}")
            coordinates = self.data_processor.read_data(file_path)

            logger.info(f"成功读取了 {len(coordinates)} 个坐标点")
            logger.info(f"成功读取数据，形状: {coordinates.shape}")
            return coordinates

        except Exception as e:
            logger.exception(f"读取文件时发生错误: {str(e)}")
            return None

    def calculate_offset_vectors(self, current_data: np.ndarray, ref_data: np.ndarray) -> np.ndarray:
        """计算每个原子位置的偏移向量

        Args:
            current_data (np.ndarray): 极化相原子坐标数组，形状为(n, 3)
            ref_data (np.ndarray): 参考相原子坐标数组，形状为(n, 3)

        Returns:
            np.ndarray: 偏移向量数组，形状为(n, 3)

        Raises:
            ValueError: 当两个数组形状不匹配时抛出
        """
        logger.info("计算偏移向量")
        if current_data.shape != ref_data.shape:
            error_msg = f"数据形状不匹配: current_data {current_data.shape} != ref_data {ref_data.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 直接使用numpy广播减法，避免循环
        return current_data - ref_data

    def calculate_angles(self, vectors: np.ndarray, axis: int = 0) -> np.ndarray:
        """计算向量在指定平面上的方向角度

        Args:
            vectors (np.ndarray): 偏移向量数组，形状为(n, 3)
            axis (int, optional): 要计算夹角的参考轴. Defaults to 0.
                                0: 在xz平面上，以x轴为参考
                                1: 在yz平面上，以y轴为参考
                                2: 在xy平面上，以x轴为参考

        Returns:
            np.ndarray: 归一化的角度值，范围[0,1]表示[0,360)度

        平面定义：
            axis = 0: xz平面
                - 0/360° -> [1,0,0]  (x轴正方向)
                - 90°    -> [0,0,1]  (z轴正方向)
                - 180°   -> [-1,0,0] (x轴负方向)
                - 270°   -> [0,0,-1] (z轴负方向)
            axis = 1: yz平面
                - 0/360° -> [0,1,0]  (y轴正方向)
                - 90°    -> [0,0,1]  (z轴正方向)
                - 180°   -> [0,-1,0] (y轴负方向)
                - 270°   -> [0,0,-1] (z轴负方向)
            axis = 2: xy平面
                - 0/360° -> [1,0,0]  (x轴正方向)
                - 90°    -> [0,1,0]  (y轴正方向)
                - 180°   -> [-1,0,0] (x轴负方向)
                - 270°   -> [0,-1,0] (y轴负方向)
        """
        if axis not in [0, 1, 2]:
            error_msg = f"无效的参考轴索引: {axis}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 定义平面映射
        plane_mapping = {
            0: ([0, 2], 'xz平面'),  # xz平面，x为参考轴
            1: ([1, 2], 'yz平面'),  # yz平面，y为参考轴
            2: ([0, 1], 'xy平面')   # xy平面，x为参考轴
        }

        # 获取平面的两个坐标轴索引
        axis_indices, plane_name = plane_mapping[axis]

        # 提取二维平面上的向量分量
        if axis == 0:
            vectors_2d = vectors[:, [0, 2]]  # xz平面
            ref_vector = np.array([1, 0])    # x轴正方向 [1,0]
            ref_vector = np.array([ref_vector]*vectors.shape[0])
        elif axis == 1:
            vectors_2d = vectors[:, [1, 2]]  # yz平面
            ref_vector = np.array([0, 1])    # y轴正方向 [1,0]
            ref_vector = np.array([ref_vector]*vectors.shape[0])
        elif axis == 2:
            vectors_2d = vectors[:, [0, 1]]  # xy平面
            ref_vector = np.array([1, 0])    # x轴正方向 [1,0]
            ref_vector = np.array([ref_vector]*vectors.shape[0])
        else:
            logger.error("错误的轴索引")
            raise ValueError("无效的轴索引")

        # 计算各向量与参考向量的弧度差
        theta_a = np.arctan2(vectors_2d[:, 1], vectors_2d[:, 0])  # y, x 顺序
        theta_b = np.arctan2(
            ref_vector[:, 1], ref_vector[:, 0])        # 参考向量角度（0弧度）
        theta = (theta_a - theta_b) % (2 * np.pi)                 # 处理负角度

        # 转换为角度并处理零向量
        angles = np.degrees(theta)
        zero_mask = (vectors_2d[:, 0] == 0) & (vectors_2d[:, 1] == 0)
        angles = angles.astype(float)
        angles[zero_mask] = np.nan

        return angles

    def plot_polarization_heatmap(self, coords: np.ndarray, offset_vectors: np.ndarray, angles: np.ndarray,
                                  proj_plane: str = 'xz', title: str = None, figsize: tuple = (10, 5),
                                  cmap: str = None, arrow_scale: float = 0.4,
                                  arrow_color: str = 'black', arrow_width: float = 0.001) -> None:
        """绘制极化热力图，包含方向箭头

        Args:
            coords (np.ndarray): 原子坐标数组，形状为(n, 3)，包含[x,y,z]坐标
            offset_vectors (np.ndarray): 偏移向量数组，形状为(n, 3)
            angles (np.ndarray): 与指定轴的夹角数组，形状为(n,)
            proj_plane (str, optional): 投影平面，可选 'xy', 'xz', 'yz'. Defaults to 'xz'.
            title (str, optional): 图表标题. Defaults to None.
            figsize (tuple, optional): 图表大小. Defaults to (10, 5).
            cmap (str, optional): 热力图颜色映射. Defaults to None.
            arrow_scale (float, optional): 箭头缩放因子. Defaults to 0.4.
            arrow_color (str, optional): 箭头颜色. Defaults to 'black'.
            arrow_width (float, optional): 箭头宽度. Defaults to 0.001.

        Returns:
            None

        Raises:
            ValueError: 当投影平面设置无效时抛出

        Examples:
            >>> # XZ平面投影
            >>> heatmap.plot_polarization_heatmap(
            ...     coords=Ti_pm3m_coord,
            ...     offset_vectors=offset_vectors,
            ...     angles=angles,
            ...     proj_plane='xz',
            ...     title="Ti原子极化分布(XZ平面)"
            ... )

            >>> # XY平面投影
            >>> heatmap.plot_polarization_heatmap(
            ...     coords=Ti_pm3m_coord,
            ...     offset_vectors=offset_vectors,
            ...     angles=angles,
            ...     proj_plane='xy',
            ...     title="Ti原子极化分布(XY平面)"
            ... )
        """
        # logger.info(f"开始绘制极化热力图 ({proj_plane.upper()}平面)")

        # 定义投影平面的坐标轴映射
        plane_mapping = {
            'xy': (0, 1, 'x', 'y'),
            'xz': (0, 2, 'x', 'z'),
            'yz': (1, 2, 'y', 'z')
        }

        if proj_plane not in plane_mapping:
            error_msg = f"无效的投影平面: {proj_plane}，应为'xy', 'xz'或'yz'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 获取投影轴的索引和标签
        axis1_idx, axis2_idx, axis1_label, axis2_label = plane_mapping[proj_plane]

        logger.info(f'angle[652]={angles[652]}')

        # 创建数据透视表，使用pm3m相的坐标
        df = pd.DataFrame({
            axis1_label: coords[:, axis1_idx],
            axis2_label: coords[:, axis2_idx],
            'angle': angles
        })

        data = df.pivot(index=axis2_label, columns=axis1_label, values='angle')

        # 创建图表
        plt.figure(figsize=figsize, dpi=300)
        ax = sns.heatmap(
            data,
            annot=False,            # 显示数值
            fmt=".2f",             # 数值格式
            cmap=self.cyclic_cmap,        # 颜色映射
            linewidths=0,        # 网格线宽
            linecolor='white',     # 网格线颜色
            cbar_kws={'label': 'Value Scale'},  # 颜色条标签
            vmin=0, vmax=360
        )

        ax.invert_yaxis()

        x_m = []
        y_m = []
        for i in range(38):
            for j in range(18):
                x_m.append(0.5+i)
                y_m.append(0.5+j)

        # 获取网格中心点的坐标
        x_centers = np.arange(len(data.columns)) + 0.5
        y_centers = np.arange(len(data.index)) + 0.5
        X, Y = np.meshgrid(x_centers, y_centers)

        try:

            U = offset_vectors[:, axis1_idx] * arrow_scale
            V = offset_vectors[:, axis2_idx] * arrow_scale
            # logger.info(f'U[652]={U[652]}')
            # logger.info(f'V[652]={V[652]}')

            # U[:5]=1
            # V[:5]=1

            # 在热力图上绘制箭头
            ax.quiver(x_m, y_m, U, V,
                      angles='xy',           # 使用xy坐标系
                      scale_units='xy',      # 使用xy单位进行缩放
                      scale=0.6,               # 缩放因子
                      color=arrow_color,
                      width=arrow_width,
                      headwidth=3,           # 箭头头部宽度
                      headlength=5,          # 箭头头部长度
                      headaxislength=4.5,    # 箭头头部轴长度
                      pivot='tail'
                      )

        except Exception as e:
            logger.error(f"绘制箭头时发生错误: {str(e)}")

        # 设置标题和样式
        if title:
            plt.title(title)
        ax.axis('off')

        # logger.info(f"热力图绘制完成")
        os.makedirs(f"./data/figure/", exist_ok=True)
        logger.info(
            f"保存热力图到: ./data/figure/{self.plane_number}_{proj_plane}.png")
        plt.savefig(f"./data/figure/{self.plane_number}_{proj_plane}.png")
        # plt.show()

    def plot_polarization_heatmap_df(self, df: pd.DataFrame, proj_plane: str = 'xz', title: str = None, figsize: tuple = (10, 5),
                                     cmap: str = None, arrow_scale: float = 0.4, arrow_color: str = 'black', arrow_width: float = 0.001,
                                     offset_vectors: np.ndarray = None) -> None:
        # 定义投影平面的坐标轴映射
        plane_mapping = {
            'xy': (0, 1, 'x', 'y'),
            'xz': (0, 2, 'x', 'z'),
            'yz': (1, 2, 'y', 'z')
        }

        if proj_plane not in plane_mapping:
            error_msg = f"无效的投影平面: {proj_plane}，应为'xy', 'xz'或'yz'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 获取投影轴的索引和标签
        axis1_idx, axis2_idx, axis1_label, axis2_label = plane_mapping[proj_plane]

        data = df.pivot(index=axis2_label, columns=axis1_label, values='angle')
        plt.figure(figsize=figsize, dpi=300)
        ax = sns.heatmap(
            data,
            annot=False,            # 显示数值
            fmt=".2f",             # 数值格式
            cmap=self.cyclic_cmap,        # 颜色映射
            linewidths=0,        # 网格线宽
            linecolor='white',     # 网格线颜色
            cbar_kws={'label': 'Value Scale'},  # 颜色条标签
            vmin=0, vmax=360
        )

        ax.invert_yaxis()

        x_m = []
        y_m = []
        for i in range(38):
            for j in range(18):
                x_m.append(0.5+i)
                y_m.append(0.5+j)

        # 获取网格中心点的坐标
        x_centers = np.arange(len(data.columns)) + 0.5
        y_centers = np.arange(len(data.index)) + 0.5
        X, Y = np.meshgrid(x_centers, y_centers)
        try:

            U = offset_vectors[:, axis1_idx] * arrow_scale
            V = offset_vectors[:, axis2_idx] * arrow_scale
            # logger.info(f'U[652]={U[652]}')
            # logger.info(f'V[652]={V[652]}')

            # U[:5]=1
            # V[:5]=1

            # 在热力图上绘制箭头
            ax.quiver(x_m, y_m, U, V,
                      angles='xy',           # 使用xy坐标系
                      scale_units='xy',      # 使用xy单位进行缩放
                      scale=0.1,               # 缩放因子
                      color=arrow_color,
                      width=arrow_width,
                      headwidth=3,           # 箭头头部宽度
                      headlength=5,          # 箭头头部长度
                      headaxislength=4.5,    # 箭头头部轴长度
                      pivot='tail'
                      )

        except Exception as e:
            logger.error(f"绘制箭头时发生错误: {str(e)}")

        # 设置标题和样式
        if title:
            plt.title(title)
        ax.axis('off')

        # logger.info(f"热力图绘制完成")
        os.makedirs(f"./data/polarization/", exist_ok=True)
        logger.info(
            f"保存热力图到: ./data/polarization/{self.plane_number}_{proj_plane}.png")
        plt.savefig(f"./data/polarization/{self.plane_number}_{proj_plane}.png")
