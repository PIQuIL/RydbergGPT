# Data

## Rydberg System
$$
\hat{H}_{\mathrm{Rydberg}} = 
\sum_{i < j}^{N} \frac{C_6}{\lVert \mathbf{r}_i - \mathbf{r}_j \rVert} \hat{n}_i \hat{n}_j - \delta \sum_{i}^{N} \hat{n}_i - \frac{\Omega}{2} \sum_{i}^{N} \hat{\sigma}_i^{(x)},
$$

$$
C_6 = \Omega \left( \frac{R_b}{a} \right)^6, \quad V_{ij} = \frac{a^6}{\lVert \mathbf{r}_i - \mathbf{r}_j \rVert^6}
$$

- $N = L \times L =$ number of atoms/qubits
- $i, j =$ qubit index
- $V_{ij} =$ blockade interaction between qubits $i$ and $j$
- $a =$ Lattice spacing
- $R_b =$ Rydberg blockade radius
- $\mathbf{r}_i =$ the position of qubit $i$
- $\hat{n}_i =$ number operator at qubit $i$
- $\delta =$ detuning at qubit $i$
- $\Omega =$ Rabi frequency at qubit $i$

## Dataset
Consider setting $\Omega = 1$ and varying the other Hamiltonian parameters independently :

- $L = [5, 6, 11, 12, 15, 16]$
- $\delta / \Omega = [-0.36, -0.13, 0.93, 1.05, 1.17, 1.29, 1.52, 1.76, 2.94, 3.17]$
- $R_b / a = [1.05, 1.15, 1.3]$
- $\beta \Omega = [0.5, 1, 2, 4, 8, 16, 32, 48, 64]$
