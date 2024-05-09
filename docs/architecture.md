# Architecture 

## Rydberg System
Consider the standard Rydberg Hamiltonian of the form :

```math
\begin{align}
& \hat{H}_{\mathrm{Rydberg}} =  \sum_{i < j} V(\lVert \mathbf{R}_i - \mathbf{R}_j \rVert \ ; R_b, \Omega) \hat{n}_i \hat{n}_j - \sum_{i} \Delta_i \hat{n}_i + \sum_{i} \frac{\Omega}{2} \sigma_i^{(x)} \\
& V(\lVert \mathbf{R}_i - \mathbf{R}_j \rVert \ ; R_b, \Omega) = \frac{R_b^6 \Omega}{\lVert \mathbf{R}_i - \mathbf{R}_j \rVert^6}
\end{align}
```

Here, $V_{ij}$ = blockade interaction strength between atoms $i$ and $j$, $R_b$ = blockade radius in units of the lattice spacing, $\hat{n}_i$ = number operator at ion $i$, $\mathbf{R}_i$ = the position of atom $i$ in units of the lattice spacing, $\Delta_i$ = detuning at atom $i$, $\Omega_i$ = Rabi frequency at atom $i$.

## Transformer

Vanilla transformer architecture taken from [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf).

![Architecture](https://github.com/PIQuIL/RydbergGPT/blob/main/resources/architecture%20diagram.jpg)

```math
\begin{align}
H_i &= \mathrm{GraphNN}(\mathrm{edges} = V_{ij} \ ; \mathrm{nodes}=\Omega_i, \Delta_i, R_b, \beta) \\
&= \text{Hamiltonian parameters encoded in a sequence by a graph neural network,} \\
\sigma_i &= \text{one-hot encoding of measured spin of qubit $i$,} \\
P_i &= P(\sigma_i | \sigma_{< i}) = \text{conditional probability distribution of spin $i$} \\
i &= \text{sequence index (either $T$ or $S$ axis shown in the architecture diagram).}
\end{align}
```

The transformer encoder represents the Rydberg Hamiltonian with a sequence. <br/>
The transformer decoder represents the corresponding ground state wavefunction.