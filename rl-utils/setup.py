"""
rl-utils
"""

from setuptools import setup

requirements = [
    'numpy',
    'pytest',
    'scipy'
]

setup(name='rl-utils',
      version='0.0.0',
      description='A library for reinforcement learning algorithms, including REPS.',
      author='Timothy Lee',
      author_email='timothyelee@cmu.edu',
      packages=['rl_utils'],
      package_dir={'': '.'},
      install_requires = requirements
      )
