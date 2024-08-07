<!-- [![Python package](https://github.com/PIQuIL/RydbergGPT/actions/workflows/python-package.yml/badge.svg)](https://github.com/PIQuIL/RydbergGPT/actions/workflows/python-package.yml) -->

# RydbergGPT
A large language model (LLM) for Rydberg atom array physics. Manuscript available on [arXiv](https://arxiv.org/abs/2405.21052).

## Table of contents
- [Quick Start](#quickstart) <br/>
    - [Configuration](#configuration) <br/>
    - [Training](#training) <br/>
- [Installation](#installation) <br/>
- [Documentation](#documentation) <br/>
- [Architecture](#architecture) <br/>
    - [Rydberg System](#rydbergsystem) <br/>
    - [Transformer](#transformer) <br/>
    - [Data](#data) <br/>


## Quick Start <a name="quickstart"></a>

### Configuration <a name="configuration"></a>
The`config.yaml` is used to define the hyperparameters for :
- Model architecture
- Training settings
- Data loading
- Others

### Training <a name="training"></a> 
To train RydbergGPT locally, execute the `main.py` with :
```bash
python main.py --config_name=config_small.yaml
```

## Installation <a name="installation"></a>
Clone the repository using the following command :
```bash
git clone https://github.com/PIQuIL/RydbergGPT
```

Create a conda environment
```bash
conda create --name rydberg_env python=3.11
```

and finally install via pip in developer mode:
```bash
cd RydbergGPT
pip install -e .
```

## Documentation <a name="documentation"></a>
Documentation is implemented with [MkDocs](https://www.mkdocs.org/) and available at https://piquil.github.io/RydbergGPT.

## Architecture  <a name="architecture"></a>

### Rydberg System <a name="rydbergsystem"></a>
Consider the standard Rydberg Hamiltonian of the form :

```math
\begin{align}
& \hat{H}_{\mathrm{Rydberg}} =  \sum_{i < j} V(\lVert \mathbf{R}_i - \mathbf{R}_j \rVert \ ; R_b, \Omega) \hat{n}_i \hat{n}_j - \sum_{i} \Delta_i \hat{n}_i + \sum_{i} \frac{\Omega}{2} \sigma_i^{(x)} \\
& V(\lVert \mathbf{R}_i - \mathbf{R}_j \rVert \ ; R_b, \Omega) = \frac{R_b^6 \Omega}{\lVert \mathbf{R}_i - \mathbf{R}_j \rVert^6}
\end{align}
```

Here, $V_{ij}$ = blockade interaction strength between atoms $i$ and $j$, $R_b$ = blockade radius in units of the lattice spacing, $\hat{n}_i$ = number operator at ion $i$, $\mathbf{R}_i$ = the position of atom $i$ in units of the lattice spacing, $\Delta_i$ = detuning at atom $i$, $\Omega_i$ = Rabi frequency at atom $i$.

### Transformer <a name="transformer"></a>

Vanilla transformer architecture taken from [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf).

<img src="https://github.com/PIQuIL/RydbergGPT/blob/main/docs/resource/architecture.png" width="600" />

- $\mathbf{x} =$ experimental settings
- $\sigma_i =$ one-hot encoding of measured qubit $i$
- $p_{\theta}(\sigma_i | \sigma_{< i}) =$ neural network conditional probability distribution of qubit $i$

The transformer encoder represents the Rydberg Hamiltonian with a sequence. <br/>
The transformer decoder represents the corresponding ground state wavefunction.

### Data <a name="data"></a>
Consider setting $\Omega = 1$ and varying the other Hamiltonian parameters independently :
```math
\begin{align}
L &= [5, 6, 11, 12, 15, 16] \\
\delta / \Omega &= [-0.36, -0.13, 0.93, 1.05, 1.17, 1.29, 1.52, 1.76, 2.94, 3.17] \\
R_b / a &= [1.05, 1.15, 1.3] \\
\beta \Omega &= [0.5, 1, 2, 4, 8, 16, 32, 48, 64]
\end{align}
```
Data available on [Pennylane Datasets](https://pennylane.ai/datasets/other/rydberggpt)
