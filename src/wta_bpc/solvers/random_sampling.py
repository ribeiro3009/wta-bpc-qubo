"""Baseline: uniform random sampling. Draw `shots` vectors y uniformly in {0,1}^K and keep the one
of lowest energy, exactly the post-processing of QAOA but without any circuit. If QAOA does not
beat this with the same number of shots, the measurements, not the circuit, are doing the work."""
import numpy as np


def solve(Q, shots=1000, seed=None):
    """Return the lowest-energy vector among `shots` uniform random 0/1 vectors."""
    K = Q.shape[0]
    if K == 0:
        return np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed)
    Y = rng.integers(0, 2, (shots, K))
    energies = np.einsum("ni,ij,nj->n", Y, Q, Y)
    return Y[np.argmin(energies)]


def solve_onehot(Q, groups, shots=1000, seed=None):
    """Baseline for the XY-mixer QAOA: in every group (the variables of one free target) pick one
    variable uniformly at random, so each sample has exactly one column per target.
    Return the lowest-energy vector among `shots` such samples."""
    K = Q.shape[0]
    rng = np.random.default_rng(seed)
    Y = np.zeros((shots, K), dtype=int)
    rows = np.arange(shots)
    for group in groups:
        chosen = np.array(group)[rng.integers(0, len(group), shots)]    # one variable per sample
        Y[rows, chosen] = 1
    energies = np.einsum("ni,ij,nj->n", Y, Q, Y)
    return Y[np.argmin(energies)]
