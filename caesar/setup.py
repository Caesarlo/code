# setup.py
from setuptools import setup, find_packages

setup(
    name='caesar',               # 包名称
    version='0.1',                         # 版本号
    packages=find_packages(),              # 自动查找包
    description='for myself',  # 描述
    author='GUMP',                    # 作者
    author_email='caesar_gump@163.com', # 作者邮箱
    install_requires=[
        # 列出依赖包，如 'numpy>=1.18.0',
    ],
)
