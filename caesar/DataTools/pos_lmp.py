import numpy as np
import dpdata
import argparse
import sys
from caesar.logger import setup_logger

# 配置日志
logger = setup_logger(__name__)


def convert_structure(input_file, output_file, direction, replicate=(1, 1, 1)):
    """
    在 VASP POSCAR 和 LAMMPS 数据格式之间转换结构。

    :param input_file: 输入文件路径。
    :param output_file: 输出文件路径。
    :param direction: 转换方向。'p2l' 或 'l2p'。
    :param replicate: 复制倍数的元组 (x, y, z)（默认值：(1, 1, 1)）。
    """
    try:
        if direction == 'p2l':
            input_format = 'vasp/poscar'
            output_format = 'lammps/lmp'
        elif direction == 'l2p':
            input_format = 'lammps/lmp'
            output_format = 'vasp/poscar'
        else:
            logger.error(
                "无效的转换方向。请选择 'p2l' 或 'l2p'。")
            return

        # 读取输入文件并创建 System 对象
        logger.info(f"正在读取 {input_format} 文件: {input_file}")
        system = dpdata.System(input_file, fmt=input_format)

        # 如果需要，进行结构复制
        if replicate != (1, 1, 1):
            logger.info(f"复制结构，倍数为: {replicate}")
            system = system.replicate(replicate)

        # 转换并写入输出格式
        logger.info(f"正在转换为 {output_format} 格式。")
        system.to(fmt=output_format, file_name=output_file)
        logger.info(f"成功转换并保存至: {output_file}")

    except Exception as e:
        logger.error(f"转换过程中发生错误: {e}")


def pos_lmp():
    parser = argparse.ArgumentParser(
        description="在 VASP POSCAR 和 LAMMPS 数据格式之间转换结构。")
    parser.add_argument('--input', '-i', type=str,
                        required=True, help='输入文件路径。')
    parser.add_argument('--output', '-o', type=str,
                        required=True, help='输出文件路径。')
    parser.add_argument('--direction', '-d', type=str, required=True,
                        choices=['p2l', 'l2p'],
                        help="转换方向：'p2l' 或 'l2p'。")
    parser.add_argument('--replicate', '-r', type=int, nargs=3, default=[1, 1, 1],
                        help='在 x, y, z 方向上的复制倍数（默认值：1 1 1）。')
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    direction = args.direction
    replicate = tuple(args.replicate)

    convert_structure(input_file, output_file, direction, replicate)


if __name__ == "__main__":
    pos_lmp()
