from setuptools import setup, find_packages
from nnTreeVB import __version__

_version = __version__

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name='nnTreeVB',
    version=_version,
    description='Neural Network-based Variational Bayesian '+\
            'Model for Phylogenetic Parameter Estimation',
    author='Amine Remita',
    packages=find_packages(),
    scripts=[
        ],
    install_requires=INSTALL_REQUIRES
)
