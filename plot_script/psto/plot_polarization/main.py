from plot_heatmap import PolarizationHeatmap
from data_utils import LatticeDataProcessor
import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from caesar.logger.logger import setup_logger

logger = setup_logger(__name__)

# 添加项目根目录到Python路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


data_processor = LatticeDataProcessor()
plot_script = PolarizationHeatmap()


@dataclass
class parameters:
    traj_file_path = "./data/200k-trajectory.xyz"
    traj_output_dir = "./data/timesteps_200k"  # 添加输出目录
    step = 50000
    lattice_output_dir = "./data/lattice_slices_200k/Ti/"
    frame_path = "./data/timesteps_200k/frame_50000.npy"  # 要处理的帧数据路径
    lattice_constant = 3.9127  # 晶格常数
    tolerance = 1.9  # 偏移
    cubic_length = 20  # 立方相的y方向扩胞数
    cubic_output_dir = "./data/cubic_402020/"
    cubic_path = './data/cubic_402020/plane_all.npy'


def process_lammps_traj(traj_file_path: str, traj_output_dir: str, step: int):
    """处理LAMMPS轨迹文件

    Args:
        traj_file_path (str): 轨迹文件路径
        traj_output_dir (str): 输出目录
        step (int): 时间步间隔
    """
    data_processor.extract_timesteps_from_lammps_traj(
        traj_file_path,  # 使用位置参数
        step,
        traj_output_dir
    )


def process_lattice_plane(frame_path: str, output_dir: str, atom_type: int, not_sort_axis: int = 1):
    frame_data = data_processor.read_data(frame_path)
    if frame_data is None:
        logger.error(f"无法读取帧数据: {frame_path}")
        return

    # 设置晶格常数（根据实际情况修改）
    data_processor.lattice_constant = parameters.lattice_constant
    data_processor.tolerance = parameters.tolerance

    data_processor.process_lattice_plane(
        data_array=frame_data,
        atom_type=atom_type,
        output_dir=output_dir,
        not_sort_axis=not_sort_axis
    )


def build_cubic_xz_plane(cubic_out_path: str, length: int, not_sort_axis: int, is_pbc: bool = False):
    # 设置晶格常数
    data_processor.lattice_constant = parameters.lattice_constant
    data_processor.tolerance = parameters.tolerance

    # 创建输出目录
    os.makedirs(os.path.dirname(cubic_out_path), exist_ok=True)

    for i in range(length):
        cubic_plane = data_processor.build_cubic_lattice(
            x=40, y=i, z=20, not_sort_axis=not_sort_axis, is_pbc=is_pbc)
        data_processor.write_data(
            data=cubic_plane,
            data_path=f"{cubic_out_path}plane_{i}.npy"
        )
        logger.info(f"生成立方相平面 {i}: {cubic_plane.shape}")


def build_cubic_all(cubic_out_path: str, is_pbc: bool = False):
    # 设置晶格常数
    data_processor.lattice_constant = parameters.lattice_constant
    data_processor.tolerance = parameters.tolerance

    # 创建输出目录
    os.makedirs(os.path.dirname(cubic_out_path), exist_ok=True)

    cubic_data = data_processor.build_cubic_lattice_all(
        x=40, y=20, z=20, is_pbc=is_pbc)
    logger.info(cubic_data.shape)
    logger.info(
        f'min_x: {min(cubic_data[:, 0])}, max_x: {max(cubic_data[:, 0])}')
    logger.info(
        f'min_y: {min(cubic_data[:, 1])}, max_y: {max(cubic_data[:, 1])}')
    logger.info(
        f'min_z: {min(cubic_data[:, 2])}, max_z: {max(cubic_data[:, 2])}')
    data_processor.write_data(
        data=cubic_data,
        data_path=f"{cubic_out_path}plane_all.npy"
    )


def calculate_offset_vector_and_plot(lattice_plane_path: str, cubic_plane_path: str):
    # 读取晶格相的平面
    logger.info(f"读取晶格相的平面: {lattice_plane_path}")
    lattice_plane = data_processor.read_data(lattice_plane_path)
    logger.info(f"晶格相的平面: {lattice_plane.shape}")
    # 读取立方相的平面
    logger.info(f"读取立方相的平面: {cubic_plane_path}")
    cubic_plane = data_processor.read_data(cubic_plane_path)
    logger.info(f"立方相的平面: {cubic_plane.shape}")
    # 修改参数名以匹配函数定义
    offset_vector = plot_script.calculate_offset_vectors(
        current_data=lattice_plane,  # 极化相数据
        ref_data=cubic_plane  # 参考相数据
    )
    logger.info(f"偏移量: {offset_vector.shape}")

    angle_deg = plot_script.calculate_angles(  # 使用plot_script的方法
        vectors=offset_vector
    )
    logger.info(f"角度: {angle_deg.shape}")

    plot_script.plot_polarization_heatmap(
        coords=cubic_plane,
        offset_vectors=offset_vector,
        angles=angle_deg,
        title=f"Ti原子极化分布(XZ平面) 截面: {plot_script.plane_number}",
        figsize=(20, 8),
        arrow_scale=1,
        arrow_color='black',
        arrow_width=0.001
    )
    logger.info('=' * 50)


def cell_coord_process(frame_path: str):
    # 设置晶格常数（根据实际情况修改）
    data_processor.lattice_constant = parameters.lattice_constant
    data_processor.tolerance = parameters.tolerance


def build_unitcell_pol_df_process(frame_path: str, cubic_coord_path: str):
    data_processor.lattice_constant = parameters.lattice_constant
    data_processor.tolerance = parameters.tolerance
    df = data_processor.build_unitcell_pol_df(
        frame_path=frame_path,
        cubic_coord_path=cubic_coord_path
    )
    # 计算角度
    vectors = df['pol'].apply(lambda x: np.array(
        x) if isinstance(x, (list, np.ndarray)) else x)
    vectors = np.array(vectors.tolist(), dtype=np.float64)
    angle = plot_script.calculate_angles(
        vectors=vectors,
        axis=0
    )
    df['angle'] = angle

    for i in range(1, 18 + 1):
        plot_data = df.loc[df['y'] == 3.9127 * i]
        offset_vectors = plot_data['pol'].apply(lambda x: np.array(
            x) if isinstance(x, (list, np.ndarray)) else x)
        offset_vectors = np.array(offset_vectors.tolist(), dtype=np.float64)
        plot_script.plane_number = i
        # logger.info(f"偏移量: {offset_vectors.shape}")
        plot_script.plot_polarization_heatmap_df(
            df=plot_data,
            proj_plane='xz',
            title=f"$(PbTiO_3)/(SrTiO_3)$极化分布(XZ平面) 截面: {i}",
            figsize=(20, 8),
            arrow_scale=1,
            arrow_color='black',
            arrow_width=0.001,
            offset_vectors=offset_vectors
        )


if __name__ == "__main__":
    # 读取lammps的traj文件
    # process_lammps_traj(parameters.traj_file_path,
    #                     parameters.traj_output_dir,
    #                     parameters.step)

    # 构建pm3m相的xz平面
    # build_cubic_xz_plane(
    #     cubic_out_path=parameters.cubic_output_dir,
    #     length=parameters.cubic_length,
    #     not_sort_axis=2,
    #     is_pbc=True
    # )

    # build_cubic_all(
    #     cubic_out_path=parameters.cubic_output_dir,
    #     is_pbc=True
    # )

    build_unitcell_pol_df_process(
        frame_path=parameters.frame_path,
        cubic_coord_path=parameters.cubic_path
    )

    # 处理每帧数据,沿y轴切面,构建xz平面
    # process_lattice_plane(
    #     frame_path=parameters.frame_path,
    #     output_dir=parameters.lattice_output_dir,
    #     atom_type=3,  # Ti原子
    #     not_sort_axis=2
    # )

    # 读取Ti原子的所有坐标，利用KD-Tree
    # cell_coord_process(parameters.frame_path)

    # 计算偏移量,绘制热力图
    # for i in range(1,20):
    #     # i=1
    #     plot_script.plane_number = i
    #     calculate_offset_vector_and_plot(
    #         lattice_plane_path=parameters.lattice_output_dir + f"plane_{i}.npy",
    #         cubic_plane_path=parameters.cubic_output_dir + f"plane_{i}.npy"
    #     )
    # break

    # a=np.array([[0,0,1],[0,1,0]])
    # b=np.array([[0,0,1],[0,0,1]])
    # angle=plot_script.calculate_angles(vectors=a)
    # print(angle)
    # break
