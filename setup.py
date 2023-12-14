"""Setup script for metareskill"""

from setuptools import setup

requirements = [
    "black",
    "hydra-core",
    "matplotlib",
    "numba",
    "numpy",
    "numpy-quaternion",
    "ray",
    "scipy",
    "scikit-learn",
    "shapely",
    "tqdm",
    "pandas",
    "squaternion",
    "mujoco-py",
    "h5py",
    "jupyterlab",
    "gym",
    "ipdb",
    "wandb",
    "icecream",
    "autolab_core",
    "sympy",
    "klampt",
    "pymdptoolbox",
    "hydra-colorlog",
    "hydra-ray-launcher",
    "patchelf",
    "protobuf==3.20.*",
    "stable_baselines3",
    "stat_utils@https://github.com/iamlab-cmu/stat-utils/archive/refs/tags/0.2.2.zip",

]

setup(
    name="recovery_skills",
    version="0.1.0",
    author="Shivam Vats",
    author_email="svats@andrew.cmu.edu",
    package_dir={"": "."},
    packages=["recovery_skills"],
    install_requires=requirements,
)
