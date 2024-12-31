from pymatgen.io.vasp import Poscar
from pymatgen.io.lammps.data import LammpsData

def poscar_to_lammps_data(poscar_file, lammps_data_file):
    # 读取 POSCAR 文件
    poscar = Poscar.from_file(poscar_file)
    structure = poscar.structure

    # 创建 LAMMPS 数据对象，atom_style atomic 需要不包含电荷等额外信息
    lammps_data = LammpsData.from_structure(structure)


    # 写入 LAMMPS 数据文件
    lammps_data.write_file(lammps_data_file)

if __name__ == "__main__":
    poscar_file = "POSCAR"          # 输入的 POSCAR 文件
    lammps_data_file = "data.lammps"   # 输出的 LAMMPS 数据文件
    poscar_to_lammps_data(poscar_file, lammps_data_file)
