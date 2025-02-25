from pymatgen.io.vasp import Poscar
from typing import List

import numpy as np
import json


def read_poscar_sites(poscar_file_path: str):
    """
    读取POSCAR文件中的原子标签和坐标
    """
    label_list = []
    coords_list = []
    poscar_data = Poscar.from_file(poscar_file_path)
    for data in poscar_data.structure.sites:
        label_list.append(data.label)
        coords_list.append(data.coords)
    return label_list, coords_list


def convert_xyz_to_xz_plane(label_list, coords_list):
    pass


def get_Ti_origin_coord(label_list, coords_list):
    lattice_constance = 3.91270
    coord_Ti_origin = []
    for i in range(3):
        for j in range(21):
            coord_Ti_origin.append(
                [lattice_constance*i, 0., lattice_constance*j])

    return coord_Ti_origin


def origin_coord(label_list, coords_list):
    """
    根据Ti周围的Sr原子，计算Ti原子的坐标
    """
    coord_Ti_list = []

    lattice_constant

    Sr_coords = []
    # 获取所有Ti原子的坐标
    for i in range(len(label_list)):
        if label_list[i] == "Ti":
            coord_Ti_list.append(coords_list[i])

    # 构建原始坐标
    for i in range(len(label_list)):
        pass

    return coord_Ti_list


def main(poscar_file_path: str):
    label_list, coords_list = read_poscar_sites(poscar_file_path)
    return label_list, coords_list


if __name__ == '__main__':
    poscar_file_path = "./data/POSCAR_sorted"
    label_list, coords_list = main(poscar_file_path)
    Ti_origin_coord_list = origin_coord(label_list, coords_list)
    with open("./data/Ti_origin_coord_list", "w", encoding='utf-8') as f:
        for i in Ti_origin_coord_list:
            f.write(f"{i}\n")
    with open('./data/Ti_coord_list', 'w', encoding='utf-8') as f:
        for i in range(len(label_list)):
            if label_list[i] == "Ti":
                f.write(f"{coords_list[i]}\n")

    # with open("./data/coords_list", "w") as f:
    #     for i in coords_list:
    #         if i == "Ti":
    #             f.write(f"{i}\n")
    # print(label_list)
    # print(coords_list)
