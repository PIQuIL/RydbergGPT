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

## References

```bib
@inproceedings{46201,
title   = {Attention is All You Need},
author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
year    = {2017},
URL = {https://arxiv.org/pdf/1706.03762.pdf}
}
```



