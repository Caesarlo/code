from Phonon import Phonon


# 初始化 Phonon 对象
phonon = Phonon('band-lmp.yaml')

# 查询特定 q-position 的数据
# q_data = phonon.get_q_position_data([0.0, 0.0, 0.0])
# print(q_data)

# 获取某种原子的所有信息
# oxygen_info = phonon.get_atom_info('O')
# print(oxygen_info)

# 导出数据为 JSON 格式
# phonon.export_to_json('phonon_data.json')

# 可视化频率分布
phonon.visualize_frequencies()



# data=phonon.get_attribute()

# print(data['lattice'])