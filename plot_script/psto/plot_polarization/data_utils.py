import numpy as np
import pandas as pd
import os
from scipy.spatial import KDTree

from caesar.logger.logger import setup_logger

logger = setup_logger(__name__)


class LatticeDataProcessor:
    """晶格数据处理器，用于处理和分析晶格结构数据

    该类提供了一系列方法来处理晶格结构数据，包括：
    1. 读取多种格式的数据文件
    2. 从LAMMPS轨迹文件中提取时间步数据
    3. 构建立方晶格结构
    4. 进行晶面分析和原子位置统计

    属性:
        lattice_constant (float): 晶格常数，用于晶格构建和分析
        current_data_path (str): 当前处理的数据文件路径（极化相）
        ref_data_path (str): 参考数据文件路径（立方相）
        tolerance (float): 晶面分析时的容差范围，默认为0.5埃

    示例:
        >>> processor = LatticeDataProcessor()
        >>> processor.lattice_constant = 3.9127  # 设置晶格常数
        >>> data = processor.read_data("structure.dat")
        >>> processor.process_lattice_plane(data, atom_type=1, output_dir="planes")
    """

    def __init__(self) -> None:
        """
        初始化
        """
        self.lattice_constant = None  # 晶格常数
        self.current_data_path = None  # 当前数据路径，即极化相文件
        self.ref_data_path = None  # 参考数据路径，即立方相文件
        self.tolerance = 0.5
        self.x = 40
        self.y = 20
        self.z = 20
        # born 有效电荷
        self.Z_A_Pb = np.array([3.74, 3.74, 3.45])
        self.Z_A_Sr = np.array([2.56, 2.56, 2.56])
        self.Z_Ti_Pb = np.array([6.17, 6.17, 5.21])
        self.Z_Ti_Sr = np.array([7.4, 7.4, 7.4])
        self.Z_O_Pb = np.array([-3.3, -3.3, -2.89])
        self.Z_O_Sr = np.array([-3.32, -3.32, -3.32])

    def build_cubic_lattice(self, x: int = 0, y: int = 0, z: int = 0, not_sort_axis: int = 1,
                            is_pbc: bool = False) -> np.ndarray:
        """构建立方晶格坐标数组

        根据给定的维度构建一个立方晶格，可以选择是否考虑周期性边界条件。
        当启用周期性边界条件时，会删除最外层的原子以避免边界效应。

        Args:
            x (int): x方向的晶格点数
            y (int): y方向的晶格点数
            z (int): z方向的晶格点数
            not_sort_axis (int, optional): 不参与排序的轴（1:x, 2:y, 3:z）. Defaults to 1.
            is_pbc (bool, optional): 是否考虑周期性边界条件. Defaults to False.

        Returns:
            np.ndarray: 形状为 (N, 3) 的坐标数组，每行表示一个晶格点的 [x,y,z] 坐标
        """
        # 修改坐标生成，从1开始
        x_coords = (np.arange(x) + 1) * self.lattice_constant
        y_coords = (np.arange(y) + 1) * self.lattice_constant
        z_coords = (np.arange(z) + 1) * self.lattice_constant

        # 根据 not_sort_axis 确定是否需要固定某个轴的值
        if not_sort_axis == 1:  # 固定 x
            # 这里x已经是指定位置，不需要+1
            x_coords = np.array([x * self.lattice_constant])
        elif not_sort_axis == 2:  # 固定 y
            # 这里y已经是指定位置，不需要+1
            y_coords = np.array([y * self.lattice_constant])
        elif not_sort_axis == 3:  # 固定 z
            # 这里z已经是指定位置，不需要+1
            z_coords = np.array([z * self.lattice_constant])

        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

        # 处理周期性边界条件
        if is_pbc and len(coords) > 0:
            # 定义边界容差
            tolerance = self.tolerance  # 使用类中定义的容差值

            # 获取坐标范围
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

            # 创建内部原子的掩码
            mask = np.ones(len(coords), dtype=bool)

            # 根据 not_sort_axis 选择要删除边界的维度
            if not_sort_axis == 1:  # 如果x轴不是固定轴，删除yz方向的边界
                mask &= ((coords[:, 1] > y_min + tolerance) &
                         (coords[:, 1] < y_max - tolerance) &
                         (coords[:, 2] > z_min + tolerance) &
                         (coords[:, 2] < z_max - tolerance)
                         )
            if not_sort_axis == 2:  # 如果y轴不是固定轴，删除xz方向的边界
                mask &= (
                        (coords[:, 0] > x_min + tolerance) &
                        (coords[:, 0] < x_max - tolerance) &
                        (coords[:, 2] > z_min + tolerance) &
                        (coords[:, 2] < z_max - tolerance)
                )

            if not_sort_axis == 3:  # 如果z轴不是固定轴，删除xy方向的边界
                mask &= ((coords[:, 0] > x_min + tolerance) &
                         (coords[:, 0] < x_max - tolerance) &
                         (coords[:, 1] > y_min + tolerance) &
                         (coords[:, 1] < y_max - tolerance))

            # 只保留内部原子
            coords = coords[mask]

        return coords

    def build_cubic_lattice_all(self,
                                x: int = 0,
                                y: int = 0,
                                z: int = 0,
                                is_pbc: bool = False) -> np.ndarray:
        """
        构建形如 x × y × z 的简单立方晶格坐标。

        当 is_pbc=True 时，删除最外层所有方向上的坐标。

        Args:
            x (int): x 方向晶格点数
            y (int): y 方向晶格点数
            z (int): z 方向晶格点数
            is_pbc (bool): 是否删除最外层边界上的坐标

        Returns:
            np.ndarray: 形状为 (N, 3) 的坐标数组，每行表示一个晶格点 [x, y, z]
        """
        # 生成网格坐标（这里从 1 开始计数）
        x_coords = (np.arange(x)) * self.lattice_constant
        y_coords = (np.arange(y)) * self.lattice_constant
        z_coords = (np.arange(z)) * self.lattice_constant

        # 创建网格
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

        # 合并为 (N, 3) 的数组
        coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

        # 如果需要删除所有方向的最外层坐标
        if is_pbc and len(coords) > 0:
            tol = self.tolerance
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

            # 同时删除 x/y/z 方向最外层
            mask = (
                (coords[:, 0] > x_min + tol) & (coords[:, 0] < x_max - tol) &
                (coords[:, 1] > y_min + tol) & (coords[:, 1] < y_max - tol) &
                (coords[:, 2] > z_min + tol) & (coords[:, 2] < z_max - tol)
            )

            coords = coords[mask]

        return coords

    @classmethod
    def extract_timesteps_from_lammps_traj(cls, input_file: str, step: int = 10000, output_dir: str = None) -> list:
        """从LAMMPS轨迹文件中提取时间步

        Args:
            input_file (str): 输入的LAMMPS轨迹文件路径
            step (int, optional): 提取时间步的间隔. Defaults to 10000.
            output_dir (str, optional): 输出目录. Defaults to None.

        Returns:
            list: 包含所有时间步的纯数据
        """
        os.makedirs(output_dir, exist_ok=True)
        all_data = []

        with open(input_file, 'r') as f_in:
            frame_count = 0
            while True:
                # 跳过原子数量行
                n_atoms_line = f_in.readline()
                if not n_atoms_line:  # 文件结束
                    logger.warning('文件结束')
                    break

                # 读取并解析Timestep行
                timestep_line = f_in.readline()
                if not timestep_line:
                    logger.warning('文件结束')
                    break

                try:
                    ts_value = int(timestep_line.strip().split()[-1])
                except (IndexError, ValueError):
                    logger.exception(f"跳过无效时间步: {timestep_line.strip()}")
                    continue

                # 验证原子数量格式
                try:
                    n_atoms = int(n_atoms_line.strip())
                except ValueError:
                    logger.exception(f"无效原子数: {n_atoms_line.strip()}")
                    continue

                # 读取原子数据块
                data_lines = []
                valid_block = True
                for _ in range(n_atoms):
                    if (line := f_in.readline()):
                        # 将字符串转换为浮点数数组
                        try:
                            values = [float(x) for x in line.strip().split()]
                            data_lines.append(values)
                        except ValueError:
                            logger.exception(f"无法解析数据行: {line.strip()}")
                            valid_block = False
                            break
                    else:
                        valid_block = False
                        break

                if not valid_block:
                    break

                # 筛选时间步
                if ts_value % step == 0:
                    # 生成纯数据文件
                    output_path = os.path.join(
                        output_dir, f"frame_{ts_value}.npy")
                    data_array = np.array(data_lines, dtype=np.float64)
                    all_data.append(data_array)

                    # 使用二进制模式写入npy文件
                    with open(output_path, 'wb') as f_out:
                        np.save(f_out, data_array)
                    logger.info(
                        f"生成文件: {output_path} (包含 {len(data_lines)} 行原子数据)")

        return np.array(all_data)

    @classmethod
    def read_data(cls, data_path: str) -> np.ndarray:
        """读取数据，支持多种格式：
        1. .npy 文件
        2. 逗号分隔的文本文件 (.csv, .txt)
        3. 空格分隔的文本文件

        Args:
            data_path (str): 数据文件路径

        Returns:
            np.ndarray: 读取的数据数组
        """
        try:
            # 获取文件扩展名
            file_ext = os.path.splitext(data_path)[1].lower()

            # 处理.npy文件
            if file_ext == '.npy':
                return np.load(data_path)

            # 处理文本文件
            with open(data_path, 'r') as f:
                # 读取第一行来判断分隔符
                first_line = f.readline().strip()
                f.seek(0)  # 重置文件指针到开始

                if ',' in first_line:
                    # 使用pandas处理逗号分隔的文件
                    data = pd.read_csv(data_path, header=None).values
                else:
                    # 处理空格分隔的文件
                    data = np.loadtxt(data_path)

                return np.array(data)

        except Exception as e:
            logger.exception(f"读取文件 {data_path} 时发生错误: {str(e)}")
            raise

    @classmethod
    def write_data(self, data: np.ndarray, data_path: str):
        """写入数据

        Args:
            data (np.ndarray): 数据
            data_path (str): 数据路径
        """
        with open(data_path, 'wb') as f:
            logger.info(f"写入数据: {data_path}")
            np.save(f, data)

    def round_to_grid(self, value):
        """将值四舍五入到最近的步长

        Args:
            value (float): 要四舍五入的值

        Returns:
            float: 四舍五入后的值
        """
        return round(value / self.lattice_constant) * self.lattice_constant

    def bucket_sort(self, data: np.ndarray, not_sort_axis: int = 1) -> np.ndarray:
        """使用桶排序对原子坐标进行排序

        Args:
            data (np.ndarray): 原子坐标数组，形状为(N, 3)，每行为[x,y,z]
            not_sort_axis (int, optional): 不参与排序的轴（1:x, 2:y, 3:z）. Defaults to 1.

        Returns:
            np.ndarray: 排序后的坐标数组
        """
        # 初始化两层桶结构
        buckets = {}

        # 根据not_sort_axis确定要排序的两个轴
        sort_axes = []
        for i in range(3):
            if i + 1 != not_sort_axis:
                sort_axes.append(i)
        primary_axis, secondary_axis = sort_axes

        # 第一次分桶：按主轴值分桶
        for point in data:
            approx_primary = self.round_to_grid(point[primary_axis])
            approx_secondary = self.round_to_grid(point[secondary_axis])

            if approx_primary not in buckets:
                buckets[approx_primary] = {}
            if approx_secondary not in buckets[approx_primary]:
                buckets[approx_primary][approx_secondary] = []
            buckets[approx_primary][approx_secondary].append(point)

        # 合并最终结果
        sorted_data = []
        # 按主轴近似值排序
        for primary_val in sorted(buckets.keys()):
            # 按次轴近似值排序
            for secondary_val in sorted(buckets[primary_val].keys()):
                sorted_data.extend(buckets[primary_val][secondary_val])

        return np.array(sorted_data)

    def lattice_slicing(self, y_coords: np.ndarray) -> np.ndarray:
        """
        晶格参数分区生成器

        Args:
            y_coords (np.ndarray): y坐标数组

        Returns:
            np.ndarray: 边界区间数组，每两个值定义一个晶面区间

        Note:
            使用self.lattice_constant作为晶格常数
            使用self.tolerance作为允许波动范围(±值)
        """
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # 计算有效晶面数量
        n_min = int(np.floor((y_min - self.tolerance) / self.lattice_constant))
        n_max = int(np.ceil((y_max + self.tolerance) / self.lattice_constant))

        # 生成晶面中心坐标
        centers = np.arange(n_min, n_max + 1) * self.lattice_constant

        # 生成边界区间
        boundaries = []
        for center in centers:
            lower = center - self.tolerance
            upper = center + self.tolerance
            # 只添加在实际坐标范围内的边界
            if lower <= y_max and upper >= y_min:
                boundaries.extend([lower, upper])

        # 确保边界是有序的且不重复
        boundaries = np.unique(boundaries)

        logger.info(f"生成了 {len(boundaries) // 2} 个晶面区间")
        return boundaries

    def process_lattice_plane(self, data_array: np.ndarray, atom_type: int, output_dir: str = None,
                              not_sort_axis: int = 1):
        """晶面提取和分析器

        将原子按晶面分层并进行统计分析。对每个晶面：
        1. 提取指定类型的原子
        2. 按照晶格常数和容差范围分层
        3. 去除最外层原子
        4. 保存每层原子的坐标数据
        5. 生成统计报告

        Args:
            data_array (np.ndarray): 原子数据数组，格式为 [type, x, y, z, ...]
            atom_type (int): 要分析的原子类型
            output_dir (str, optional): 输出目录路径. Defaults to None.
            not_sort_axis (int, optional): 不排序的轴(x:1,y:2,z:3). Defaults to 1.

        Returns:
            None
        """
        # 筛选目标原子
        filtered = data_array[data_array[:, 0] == atom_type]
        if len(filtered) == 0:
            logger.warning(f"未找到类型 {atom_type} 的原子")
            return

        y_coords = filtered[:, 2]  # y坐标

        # 生成晶格分区
        bin_edges = self.lattice_slicing(y_coords)
        num_slices = len(bin_edges) // 2  # 每个晶面对应两个边界

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存每个晶面
        results = []
        for i in range(num_slices):
            y_start = bin_edges[2 * i]
            y_end = bin_edges[2 * i + 1]

            # 筛选区间原子
            mask = (y_coords >= y_start) & (y_coords <= y_end)
            slice_data = filtered[mask]

            # 创建内部原子的掩码
            mask_cut = np.ones(len(slice_data), dtype=bool)

            # 保存完整的坐标数据 [x, y, z]
            coords_data = slice_data[:, 1:4]
            # logger.info(f'{coords_data.shape}')
            # # 获取坐标范围
            # x_min, x_max = coords_data[:, 0].min(), coords_data[:, 0].max()
            # y_min, y_max = coords_data[:, 1].min(), coords_data[:, 1].max()
            # z_min, z_max = coords_data[:, 2].min(), coords_data[:, 2].max()
            x_min, x_max = self.lattice_constant, self.lattice_constant * self.x
            y_min, y_max = self.lattice_constant, self.lattice_constant * self.y
            z_min, z_max = self.lattice_constant, self.lattice_constant * self.z

            # logger.info(f'x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}, z_min: {z_min}, z_max: {z_max}')
            # 去除最外层原子
            if not_sort_axis == 1:  # 如果x轴不是固定轴，删除yz方向的边界
                mask_cut &= ((coords_data[:, 1] > y_min + self.tolerance) &
                             (coords_data[:, 1] < y_max - self.tolerance) &
                             (coords_data[:, 2] > z_min + self.tolerance) &
                             (coords_data[:, 2] < z_max - self.tolerance))

            if not_sort_axis == 2:  # 如果y轴不是固定轴，删除xz方向的边界
                mask_cut &= ((coords_data[:, 0] > x_min + self.tolerance) &
                             (coords_data[:, 0] < x_max - self.tolerance) &
                             (coords_data[:, 2] > z_min + self.tolerance) &
                             (coords_data[:, 2] < z_max - self.tolerance))

            if not_sort_axis == 3:  # 如果z轴不是固定轴，删除xy方向的边界
                mask_cut &= ((coords_data[:, 0] > x_min + self.tolerance) &
                             (coords_data[:, 0] < x_max - self.tolerance) &
                             (coords_data[:, 1] > y_min + self.tolerance) &
                             (coords_data[:, 1] < y_max - self.tolerance))

            # # 只保留内部原子
            coords_data = coords_data[mask_cut]
            # 生成文件名
            filename = f"plane_{i}.npy"
            filepath = os.path.join(output_dir, filename)

            # 排序保证与生成的pm3m坐标一致，使用桶排序
            coords_data = self.bucket_sort(coords_data, not_sort_axis)
            # 保存数据
            np.save(filepath, coords_data)
            # logger.info(f"保存数据: {coords_data}")

            # 记录统计信息
            results.append({
                "plane": i,
                "y_range": (y_start, y_end),
                "atom_count": len(coords_data),
                "x_mean": coords_data[:, 0].mean(),
                "y_mean": coords_data[:, 1].mean(),
                "z_mean": coords_data[:, 2].mean()
            })

        # 输出统计报告
        logger.info("\n晶面分布分析：")
        logger.info(
            f"{'晶面':<8} {'Y范围':<10} {'原子数':<7} {'X平均':<7} {'Y平均':<7} {'Z平均':<7}")
        for r in results:
            logger.info(f"{r['plane']:03d}  [{r['y_range'][0]:<7.3f}, {r['y_range'][1]:>7.3f}]  "
                        f"{r['atom_count']:<8}  {r['x_mean']:.3f}    {r['y_mean']:.3f}    {r['z_mean']:.3f}")

    def process_A_coord(self, raw: list):
        # 根据中心点坐标[0.5,0.5,0.5]处理A位原子, 共八个顶角, [0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]
        A_coord = [[0, 0, 0]] * 8
        # [0,0,0]
        A_coord[0] = raw[0] - self.lattice_constant / 2, \
            raw[1] - self.lattice_constant / 2, \
            raw[2] - self.lattice_constant / 2
        # [0,0,1]
        A_coord[1] = raw[0] - self.lattice_constant / 2, \
            raw[1] - self.lattice_constant / 2, \
            raw[2] + self.lattice_constant / 2
        # [0,1,0]
        A_coord[2] = raw[0] - self.lattice_constant / 2, \
            raw[1] + self.lattice_constant / 2, \
            raw[2] - self.lattice_constant / 2
        # [0,1,1]
        A_coord[3] = raw[0] - self.lattice_constant / 2, \
            raw[1] + self.lattice_constant / 2, \
            raw[2] + self.lattice_constant / 2
        # [1,0,0]
        A_coord[4] = raw[0] + self.lattice_constant / 2, \
            raw[1] - self.lattice_constant / 2, \
            raw[2] - self.lattice_constant / 2
        # [1,0,1]
        A_coord[5] = raw[0] + self.lattice_constant / 2, \
            raw[1] - self.lattice_constant / 2, \
            raw[2] + self.lattice_constant / 2
        # [1,1,0]
        A_coord[6] = raw[0] + self.lattice_constant / 2, \
            raw[1] + self.lattice_constant / 2, \
            raw[2] - self.lattice_constant / 2
        # [1,1,1]
        A_coord[7] = raw[0] + self.lattice_constant / 2, \
            raw[1] + self.lattice_constant / 2, \
            raw[2] + self.lattice_constant / 2
        # 将A_coord转换为numpy数组
        A_coord = np.array(A_coord)
        return A_coord

    def process_O_coord(self, raw: list):
        # 根据中心点坐标[0.5,0.5,0.5]处理O位原子, 共六个面心, [0.5, 0.5, 0], [0.5, 0.5, 1], [0.5, 0, 0.5], [0.5, 1, 0.5], [0, 0.5, 0.5], [1, 0.5, 0.5]
        O_coord = [[0, 0, 0]] * 6
        # [0.5, 0.5, 0]
        O_coord[0] = raw[0], raw[1], raw[2] - self.lattice_constant / 2
        # [0.5, 0.5, 1]
        O_coord[1] = raw[0], raw[1], raw[2] + self.lattice_constant / 2
        # [0.5, 0, 0.5]
        O_coord[2] = raw[0], raw[1] - self.lattice_constant / 2, raw[2]
        # [0.5, 1, 0.5]
        O_coord[3] = raw[0], raw[1] + self.lattice_constant / 2, raw[2]
        # [0, 0.5, 0.5]
        O_coord[4] = raw[0] - self.lattice_constant / 2, raw[1], raw[2]
        # [1, 0.5, 0.5]
        O_coord[5] = raw[0] + self.lattice_constant / 2, raw[1], raw[2]
        # 将O_coord转换为numpy数组
        O_coord = np.array(O_coord)

        return O_coord

    def cal_polarization(self, A_coord: np.ndarray, Ti_coord: np.ndarray, O_coord: np.ndarray, A_cubic: np.ndarray,
                         Ti_cubic: np.ndarray, O_cubic: np.ndarray, volume: float, atom_coord: np.ndarray, kdtree: KDTree):
        A_label = [int(i) for i in A_coord[:, 0]]
        label_unique = np.unique(A_label)

        if len(label_unique) == 2:
            A_offset = np.array(A_coord[:, [1, 2, 3]] - A_cubic)
            Ti_offset = np.array(Ti_coord[[1, 2, 3]] - Ti_cubic)
            O_offset = np.array(O_coord[:, [1, 2, 3]] - O_cubic)

            Ti_pol = Ti_offset*self.Z_Ti_Pb
            # 判断O原子的位置，这里没有完善，是STO占主导还是PTO
            O_pol = np.sum(O_offset * self.Z_O_Sr, axis=0) / 2

            A_pol = []
            for i in range(len(A_coord)):
                if A_coord[i][0] == 1:
                    A_pol.append(A_offset[i]*self.Z_A_Pb)
                else:
                    A_pol.append(A_offset[i]*self.Z_A_Sr)
            A_pol = np.sum(A_pol, axis=0) / 8
            pol = (A_pol + Ti_pol + O_pol) / volume
            # logger.info(f'pol: \n{pol}')
        elif len(label_unique) == 1:
            A_offset = np.array(A_coord[:, [1, 2, 3]] - A_cubic)
            Ti_offset = np.array(Ti_coord[[1, 2, 3]] - Ti_cubic)
            O_offset = np.array(O_coord[:, [1, 2, 3]] - O_cubic)
            if label_unique == 2:
                A_pol = np.sum(A_offset * self.Z_A_Sr, axis=0) / 8
                Ti_pol = Ti_offset * self.Z_Ti_Sr
                O_pol = np.sum(O_offset * self.Z_O_Sr, axis=0) / 2

                pol = (A_pol + Ti_pol + O_pol) / volume
                # logger.info(f'pol: \n{pol}')
            if label_unique == 1:
                A_pol = np.sum(A_offset * self.Z_A_Pb, axis=0) / 8
                Ti_pol = Ti_offset * self.Z_Ti_Pb
                O_pol = np.sum(O_offset * self.Z_O_Pb, axis=0) / 2

                pol = (A_pol + Ti_pol + O_pol) / volume
                # logger.info(f'pol: \n{pol}')

        else:
            logger.warning('A_label 中存在未知标签')
            raise ValueError('A_label 中存在未知标签')

        return pol

    def build_unitcell_pol_df(self, frame_path: str, cubic_coord_path: str):
        atom_coord = self.read_data(frame_path)
        cubic_coord = self.read_data(cubic_coord_path)

        x_min, x_max = min(atom_coord[:, 1]), max(atom_coord[:, 1])
        y_min, y_max = min(atom_coord[:, 2]), max(atom_coord[:, 2])
        z_min, z_max = min(atom_coord[:, 3]), max(atom_coord[:, 3])
        x_avg = (x_max - x_min) / 40
        y_avg = (y_max - y_min) / 20
        z_avg = (z_max - z_min) / 20
        volume = x_avg * y_avg * z_avg

        # logger.info(f'atom_coord: \n{atom_coord[:, [1, 2, 3][:5]]}')
        kdtree = KDTree(atom_coord[:, [1, 2, 3]])

        df = pd.DataFrame(columns=['x', 'y', 'z', 'pol'])

        for raw in cubic_coord:
            A_cubic = self.process_A_coord(raw)
            Ti_cubic = raw
            O_cubic = self.process_O_coord(raw)
            _, A_index = kdtree.query(A_cubic)
            _, Ti_index = kdtree.query(Ti_cubic)
            _, O_index = kdtree.query(O_cubic)
            A_coord = atom_coord[A_index]
            Ti_coord = atom_coord[Ti_index]
            O_coord = atom_coord[O_index]
            # logger.info(f'A_coord: \n{A_coord}')
            # logger.info(f'Ti_coord: \n{Ti_coord}')
            # logger.info(f'O_coord: \n{O_coord}')

            pol = self.cal_polarization(A_coord, Ti_coord, O_coord,
                                        A_cubic, Ti_cubic, O_cubic, volume, atom_coord, kdtree)
            df.loc[len(df)] = [raw[0], raw[1], raw[2], pol]

            # break
        return df
