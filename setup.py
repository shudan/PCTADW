import os

from setuptools import setup

setup(
    name = 'pctadw',
    description = 'Embeddings of directed networks with text-associated nodes.',
    author = 'Shudan Zhong',
    packages=[
        'pctadw',
    ],
    entry_points={'console_scripts': ['pctadw = pctadw.__main__:main']},
    license='GPLv3 or later',
    package_dir={'pctadw': 'pctadw'},
    include_package_data=True
)

