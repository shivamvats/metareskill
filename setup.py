"""Setup script for metareskill"""

from setuptools import setup

requirements = [
    'black',
    'hydra-core',
    'matplotlib',
    'numba',
    'numpy',
    'numpy-quaternion',
    'ray',
    'scipy',
    'sklearn',
    'shapely',
    'tqdm',
    'pandas',
    'squaternion',
    'mujoco-py',
    'h5py',
    'jupyterlab',
    'gym',
    'ipdb',
    'wandb',
    'icecream',
    'autolab_core',
    'sympy',
    'klampt',
    'pymdptoolbox',
]

setup(name='recovery_skills',
        version='0.1.0',
        author='Shivam Vats',
        author_email='svats@andrew.cmu.edu',
        package_dir = {'': '.'},
        packages=['recovery_skills'],
        install_requires=requirements
        )
