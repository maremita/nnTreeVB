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
        "scripts/nntreevb.py",
        "scripts/nntreevb_write_run_exps.py",
        "scripts/nntreevb_sum_plot_exps.py",
        "scripts/slurm_exps.sh",
        "scripts/uncompress_exps_pkl.py"
        ],
    install_requires=INSTALL_REQUIRES
)
