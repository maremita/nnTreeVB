# nnTreeVB: a Neural Network-based Variational Bayesian Model for Phylogenetic Parameter Estimation

#### *Prior Density Learning in VBPPI* paper
The configuration files of the experiments performed in the paper [**Prior Density Learning in Variational Bayesian Phylogenetic Parameters Inference**](https://arxiv.org/abs/2302.02522) can be found from this link [https://github.com/maremita/nnTreeVB_Exp/tree/main/exp_learn_priors](https://github.com/maremita/nnTreeVB_Exp/tree/main/exp_learn_priors).

#### Disclaimer
**nnTreeVB** is actively in the development phase. The names of the different package entities and the default values are subject to change.
Please, feel free to contact me if you want to refactor, add, or discuss a feature.

## Overview
**nnTreeVB** is a deep variationl model that simultaneously estimates evolutionary parameters using a phylogenetic tree and multiple sequence alignment

## Dependencies
The `nnTreeVB` package depends on Python packages that could be installed with `pip` and `conda`.
The main packages that `nnTreeVB` uses are `pytorch`, `numpy` and `biopython` and `ete3`. 
You can find the complete list of dependencies in the `requirements.txt` file. 
This file is used automatically by `setup.py` to install `nnTreeVB` using `pip` or `conda`.


## Installation guide
`nnTreeVB` is developed in Python3 and can be easily installed using `pip`. I recommend installing the package in a separate virtual environment (`virtualenv`, `venv`, `conda env`, ect.). 
I haven't tested the installation yet using `conda`.

Once the virtual environment is created, `nnTreeVB` can be installed from the git repository directly through `pip`:
```
python -m pip install git+https://github.com/maremita/nnTreeVB.git
```
or by cloning the repository  and installing with `pip`:
```
git clone https://github.com/maremita/nnTreeVB.git
cd nnTreeVB
pip install .
```

## Usage

### Using `nnTreeVB` package
Classes and functions implemented in the `nnTreeVB` package can be called and used in Python scripts and notebooks.
This is an example of how to use `nnTreeVB` to fit an `nnTreeVB_GTR` to estimate branch lengths, substitution rates and relative frequencies.

```python


```

## Experiments


## How to cite
I am preparing the main `nnTreeVB` manuscript. In the meantime, if you want to refer to the framework, you can cite this preprint:

```
@article{remita2023learn_vbprior,
  title={Prior Density Learning in Variational Bayesian Phylogenetic Parameters Inference},
  author={Remita, Amine M and Diallo, Golrokh Kiani Vitae, Abdoulaye Banir{\'e}},
  journal={arXiv preprint arXiv:2302.02522},
  year={2023}
}
```

## License
The nnTreeVB package including the modules and the scripts is distributed under the **MIT License**.


## Contact
If you have any questions, please do not hesitate to contact:
- Amine Remita <remita.amine@courrier.uqam.ca>
