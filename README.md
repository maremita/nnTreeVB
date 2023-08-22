# nnTreeVB: a Neural Network-based Variational Bayesian Framework for Phylogenetic Parameter Estimation

#### *Prior Density Learning in VBPPI* paper
The configuration files of the experiments performed in the paper [Prior Density Learning in Variational Bayesian Phylogenetic Parameters Inference](https://arxiv.org/abs/2302.02522) can be found in the repository [nnTreeVB\_Exp](https://github.com/maremita/nnTreeVB_Exp/tree/main/exp_learn_priors).

[nntreevb\_learn\_prior.ipynb](https://github.com/maremita/nnTreeVB/blob/main/notebooks/nntreevb_learn_prior.ipynb) is an example of using `nnTreeVB` to learn prior densities of branch lengths with a JC69 substitution model. 

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
Experiments for the assessment of `nnTreeVB` can be found in the project [nnTreeVB\_Exp](https://github.com/maremita/nnTreeVB_Exp/).

## How to cite
I am preparing the main `nnTreeVB` manuscript. In the meantime, if you want to refer to the framework, you can cite this preprint:

```
@InProceedings{remita2023learn_vbprior,
	author={Remita, Amine M. and Vitae, Golrokh and Diallo, Abdoulaye Banir{\'e}},
	editor={Jahn, Katharina and Vina{\v{r}}, Tom{\'a}{\v{s}}},
	title={Prior Density Learning in Variational Bayesian Phylogenetic Parameters Inference},
	booktitle={Comparative Genomics},
	year={2023},
	publisher={Springer Nature Switzerland},
	address={Cham},
	pages={112--130},
	isbn={978-3-031-36911-7},
	doi={10.1007/978-3-031-36911-7_8}
}
```

## License
The nnTreeVB package including the modules and the scripts is distributed under the **MIT License**.


## Contact
If you have any questions, please do not hesitate to contact:
- Amine Remita <remita.amine@courrier.uqam.ca>
