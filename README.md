[![Python package](https://github.com/PIQuIL/RydbergGPT/actions/workflows/python-package.yml/badge.svg)](https://github.com/PIQuIL/RydbergGPT/actions/workflows/python-package.yml)

# RydbergGPT
A large language model (LLM) for Rydberg atom array physics.

## Architecture

Vanilla transformer architecture taken from [Attention is All You Need](https://research.google/pubs/pub46201/).

![Architecture](https://github.com/PIQuIL/RydbergGPT/blob/main/resources/architecture%20diagram.jpg)

```math
\begin{align}
H_i &= (\Omega_i, \Delta_i, R^{x}_i, R^{y}_i) = \text{Hamiltonian parameters of qubit $i$,} \\
\sigma_i &= \text{one-hot encoding of measured spin of qubit $i$,} \\
P_i &= P(\sigma_i | \sigma_{< i}) = \text{conditional probability distribution of spin $i$} \\
i &= \text{sequence index (either $T$ or $S$ axis shown in the architecture diagram).}
\end{align}
```

The transformer encoder encodes the Rydberg Hamiltonian into a sequential latent space. \\
The transformer decoder encodes a ground state wavefunction based on the encoded Rydberg Hamiltonian.

## Installation

## Documentation


## Acknowledgements

We found several very helpful codebases when building this repo, and we sincerely thank their authors:

+ Transformer Tutorials:
    + [Annotated Transformer](https://github.com/harvardnlp/annotated-transformer/)
+ Transformer quantum state:
    + [Predicting Properties of Quantum Systems with Conditional Generative Models](https://github.com/PennyLaneAI/generative-quantum-states)
    + [Transformer Quantum State](https://github.com/yuanhangzhang98/transformer_quantum_state)


## References

```bib
@inproceedings{46201,
title   = {Attention is All You Need},
author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
year    = {2017},
URL = {https://arxiv.org/pdf/1706.03762.pdf}
}
```
