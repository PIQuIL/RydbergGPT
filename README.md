# RydbergGPT
A large language model (LLM) for Rydberg atom physics.

## Architecture

Vanilla transformer architecture taken from [Attention is All You Need](https://research.google/pubs/pub46201/).

![Architecture](https://github.com/PIQuIL/RydbergGPT/blob/main/resources/architecture%20diagram.jpg)

```math
H_i = (\Omega_i, \Delta_i, R^{x}_i, R^{y}_i) = Hamiltonian parameters for spin i \sigma_i = one-hot encoding of measured state of spin i
P_i = P(\sigma_i | \sigma_{<i}) = conditional probabilities
```

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



