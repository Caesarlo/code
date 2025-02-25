import numpy as np
import yaml
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy.typing as npt
import seaborn as sns

from MDAnalysis import Universe
from dataclasses import dataclass
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from matplotlib import gridspec
from caesar.logger.logger import setup_logger
from pathlib import Path
from typing import List, Dict, Any, Tuple
from matplotlib import rcParams


config = {
    "font.family": 'serif',
    # "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


logger = setup_logger(__name__)


def read_frame(file_path):
    with open(file_path, 'r') as f:
        logger.info(f"读取文件: {file_path}")
        data = [[float(i) for i in line.strip().split()]
                for line in f.readlines()]
    return np.array(data)


def lattice_slicing(frame_data, lattice_constant=3.9127, tolerance=0.1):
    pass


if __name__ == '__main__':
    frame_data = read_frame('./data/frames/frame_0.dat')
    print(frame_data)
