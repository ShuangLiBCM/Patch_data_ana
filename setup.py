#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='ML_projects',
    version='0.0.0',
    description='Next generation learning plasticity model',
    author='Shuang Li',
    author_email='shuang.li@bcm.edu',
    url='https://github.com/ShuangLiBCM/learning_plasticity',
    packages=find_packages(exclude=[]),
    install_requires=['numpy'],
)

setup(
    name="sqlite3_kernel",
    version = "1.0",
    packages=find_packages(),
    description="SQLite3 Jupyter Kernel",
    url = "https://github.com/brownan/sqlite3_kernel",
    classifiers = [
        'Framework :: IPython',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: SQL',
    ]
)