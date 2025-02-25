import numpy as np
import os

from caesar.logger.logger import setup_logger

logger = setup_logger(__name__)

# 构建原始坐标


def build_pm3m_coord_all():
    pm3m_Ti_coord = []
    lattice_constance = 3.9127
    for i in range(40):
        for j in range(20):
            for k in range(20):
                pm3m_Ti_coord.append(
                    [lattice_constance*i, lattice_constance*j, lattice_constance*k])

    return np.array(pm3m_Ti_coord)

def build_pm3m_coord_y0(y0):
    pm3m_Ti_coord = []
    lattice_constance = 3.9127
    for i in range(40):
        for j in range(20):
            pm3m_Ti_coord.append(
                [lattice_constance*i, y0, lattice_constance*j])

    return np.array(pm3m_Ti_coord)

def extract_timesteps(input_file, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r') as f_in:
        frame_count = 0
        while True:
            # 跳过原子数量行
            n_atoms_line = f_in.readline()
            if not n_atoms_line:  # 文件结束
                logger.error('文件结束')
                break

            # 读取并解析Timestep行
            timestep_line = f_in.readline()
            if not timestep_line:
                logger.error('文件结束')
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
                if (data_line := f_in.readline()):
                    data_lines.append(data_line)
                else:
                    valid_block = False
                    break

            if not valid_block:
                break

            # 筛选时间步
            if ts_value % 10000 == 0:
                # 生成纯数据文件
                output_path = os.path.join(output_dir, f"frame_{ts_value}.dat")
                with open(output_path, 'w') as f_out:
                    f_out.writelines(data_lines)
                logger.info(
                    f"生成文件: {output_path} (包含 {len(data_lines)} 行原子数据)")


def read_frame_data(frame_path):
    with open(frame_path, 'r') as f_in:
        data_lines = [[float(j) for j in i.strip().split()]
                      for i in f_in.readlines()]
    return np.array(data_lines)


def lattice_slicing(y_coords, lattice_constant=3.9127, tolerance=0.5):
    """
    晶格参数分区生成器
    参数：
    y_coords: y坐标数组
    lattice_constant: 晶格常数 (Å)
    tolerance: 允许波动范围 (±值)
    """
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # 计算有效晶面数量
    n_min = int(np.floor((y_min - tolerance) / lattice_constant))
    n_max = int(np.ceil((y_max + tolerance) / lattice_constant))

    # 生成晶面中心坐标
    centers = np.arange(n_min, n_max+1) * lattice_constant

    # 生成边界区间
    bin_edges = np.sort(np.concatenate([
        centers - tolerance,
        centers + tolerance
    ]))

    # 限制在实际坐标范围内
    return np.clip(bin_edges, y_min, y_max)


def process_lattice_plane(data_array, atom_type=3, output_dir="./lattice_slices"):
    """
    晶面提取，Ti原子晶面
    """
    # 筛选目标原子
    filtered = data_array[data_array[:, 0] == atom_type]
    if len(filtered) == 0:
        logger.warning(f"未找到类型 {atom_type} 的原子")
        return

    y_coords = filtered[:, 2]

    # 生成晶格分区
    bin_edges = lattice_slicing(y_coords)
    num_slices = len(bin_edges) // 2  # 每个晶面对应两个边界

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存每个晶面
    results = []
    for i in range(num_slices):
        y_start = bin_edges[2*i]
        y_end = bin_edges[2*i+1]

        # 筛选区间原子
        mask = (y_coords >= y_start) & (y_coords <= y_end)
        slice_data = filtered[mask]

        # 生成文件名
        filename = f"plane_{i:02d}_y_{y_start:.3f}-{y_end:.3f}.dat"
        filepath = os.path.join(output_dir, filename)

        # 保存完整坐标数据
        np.savetxt(filepath,
                   slice_data[:, 1:4],  # x,y,z 三列
                   fmt='%.5f',
                   header=f"Lattice plane {i}\nY range: [{y_start:.3f}, {y_end:.3f}]\nX\tY\tZ",
                   comments='')

        # 记录统计信息
        results.append({
            "plane": i,
            "y_range": (y_start, y_end),
            "atom_count": len(slice_data),
            "x_mean": slice_data[:, 1].mean(),
            "y_mean": slice_data[:, 2].mean(),
            "z_mean": slice_data[:, 3].mean()
        })

    # 输出统计报告
    logger.info("\n晶面分布分析：")
    logger.info(
        f"{'晶面':<6} {'Y范围':<25} {'原子数':<8} {'X平均':<10} {'Y平均':<10} {'Z平均':<10}")
    for r in results:
        logger.info(f"{r['plane']:03d}  [{r['y_range'][0]:<7.3f}, {r['y_range'][1]:>7.3f}]  "
                    f"{r['atom_count']:<8}  {r['x_mean']:.3f}    {r['y_mean']:.3f}    {r['z_mean']:.3f}")


if __name__ == "__main__":
    pm3m_Ti_coord_y0 = build_pm3m_coord_y0(0.000)

    with open('./data/pm3m_Ti_coord_y0.npy', 'wb') as f:
        np.save(f, pm3m_Ti_coord_y0)

    # xyzfile = './data/1-trajectory.xyz'

    # extract_timesteps(xyzfile, "./data/frames/")

    # with open('./data/frames/frame_0.dat', 'r') as f_in:
    #     data_lines = [[float(j) for j in i.strip().split()]
    #                   for i in f_in.readlines()]
    # data_frame = np.array(data_lines)

    # data_frame = np.array(read_frame_data('./data/frames/frame_0.dat'))

    # print(data_frame)

    process_lattice_plane(data_frame,atom_type=3)
