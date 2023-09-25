<!-- [![Python package](https://github.com/PIQuIL/RydbergGPT/actions/workflows/python-package.yml/badge.svg)](https://github.com/PIQuIL/RydbergGPT/actions/workflows/python-package.yml) -->

# RydbergGPT
A large language model (LLM) for Rydberg atom array physics.

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
- [Acknowledgements](#acknowledgements) <br/>
- [References](#references) <br/>


## Quick Start <a name="quickstart"></a>

### Configuration <a name="configuration"></a>
The`config.yaml` is used to define the hyperparameters for :
1. Model architecture
2. Training settings
3. Data loading
4. Others

### Training <a name="training"></a> 
To train RydbergGPT locally, execute the `train.py` with :
```bash
python train.py --config_name=config_small.yaml
```
For the cluster:
```bash
.scripts/train.sh
```

## Installation <a name="installation"></a>
Clone the repository using the following command :
```bash
git clone https://github.com/PIQuIL/RydbergGPT
```
Install with pip :
```bash
cd RydbergGPT
pip install .
```


## Documentation <a name="documentation"></a>

Currently unavaiable

## Architecture  <a name="architecture"></a>


### Rydberg System <a name="rydbergsystem"></a>
Consider the standard Rydberg Hamiltonian of the form :

```math
\begin{align}
\hat{H}_{\mathrm{Rydberg}} =  \sum_{i < j} V(\lVert \mathbf{R}_i - \mathbf{R}_j \rVert ; R_b) \hat{n}_i \hat{n}_j - \sum_{i} \Delta_i \hat{n}_i - \sum_{i} \frac{\Omega}{2} \sigma_i^{(x)}
\end{align}
```

Here, $V_{ij}$ = blockade interaction strength between atoms $i$ and $j$, $R_b$ = blockade radius, $\hat{n}_i$ = number operator at ion $i$, $\mathbf{R}_i$ = the position of atom $i$, $\Delta_i$ = detuning at atom $i$, $\Omega_i$ = Rabi frequency at atom $i$.

### Transformer <a name="transformer"></a>

Vanilla transformer architecture taken from [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf).

![Architecture](https://github.com/PIQuIL/RydbergGPT/blob/main/resources/architecture%20diagram.jpg)

```math
\begin{align}
H_i &= \mathrm{GraphNN}(\mathrm{edges} = V_{ij}; \mathrm{nodes}=\Omega_i, \Delta_i, R_b, \beta) = \text{Hamiltonian parameters encoded in a sequence by a graph neural network,} \\
\sigma_i &= \text{one-hot encoding of measured spin of qubit $i$,} \\
P_i &= P(\sigma_i | \sigma_{< i}) = \text{conditional probability distribution of spin $i$} \\
i &= \text{sequence index (either $T$ or $S$ axis shown in the architecture diagram).}
\end{align}
```

The transformer encoder represents the Rydberg Hamiltonian with a sequence. <br/>
The transformer decoder represents the corresponding ground state wavefunction.

### Data <a name="data"></a>
Consider setting $\Omega = 4.24$ and varying the other Hamiltonian parameters independently :
```math
\begin{align}
L &= [5, 6, 11, 12, 15, 16, 19, 20] \\
\Delta &= [-1.545, -0.545, 3.955, 4.455, 4.955, 5.455, 6.455, 7.455, 12.455, 13.455] \\
R_b &= [1.05, 1.15, 1.3] \\
\beta &= [0.5, 1, 2, 4, 8, 16, 32, 48, 64]
\end{align}
```
There are a total of `8 x 10 x 3 x 9 = 2160` configurations (see [table](https://github.com/PIQuIL/RydbergGPT/blob/main/resources/Generated_training_data.md)).

## Acknowledgements <a name="acknowledgements"></a>

We found several very helpful codebases when building this repo, and we sincerely thank their authors:

+ Transformer Tutorials:
    + [Annotated Transformer](https://github.com/harvardnlp/annotated-transformer/)
    + [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
+ Transformer quantum state:
    + [Predicting Properties of Quantum Systems with Conditional Generative Models](https://github.com/PennyLaneAI/generative-quantum-states)
    + [Transformer Quantum State](https://github.com/yuanhangzhang98/transformer_quantum_state)


## References <a name="references"></a>

```bib
@inproceedings{46201,
title	= {Attention is All You Need},
author	= {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
year	= {2017},
URL	= {https://arxiv.org/pdf/1706.03762.pdf}
}
```
