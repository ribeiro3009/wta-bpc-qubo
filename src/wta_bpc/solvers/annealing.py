"""Simulated annealing for QUBO: the classical reference heuristic.

Start from a random y at high temperature T; repeatedly try flipping one bit, accept if the
energy goes down, or with probability exp(-dE / T) if it goes up; lower T geometrically.
Hot: escapes local minima. Cold: pure descent. Several independent runs, keep the best y seen.
"""
import math

import numpy as np


def solve(Q, num_reads=10, num_sweeps=500, seed=None):
    """Return the best 0/1 vector found for min y^T Q y (Q upper triangular or symmetric).

    num_reads: independent runs; num_sweeps: passes over all K bits per run."""
    K = Q.shape[0]
    if K == 0:
        return np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed)

    # E(y) = sum_k d_k y_k + sum_{k<l} S_kl y_k y_l with S symmetric, zero diagonal.
    d = np.diag(Q).copy()
    S = np.triu(Q, 1) + np.tril(Q, -1).T
    S = S + S.T
    # Flipping bit k changes E by (1 - 2 y_k) * (d_k + field_k), where field = S @ y.

    # Temperatures: from about the largest possible energy change down to 1/1000 of it.
    t_hot = max(np.abs(d).max(), np.abs(S).sum(axis=1).max())
    temperatures = t_hot * np.geomspace(1.0, 1e-3, num_sweeps)

    best_y, best_e = None, np.inf
    for _ in range(num_reads):
        y = rng.integers(0, 2, K)
        field = S @ y
        e = float(d @ y + y @ np.triu(S, 1) @ y)
        if e < best_e:
            best_y, best_e = y.copy(), e
        for T in temperatures:
            for k in rng.permutation(K):
                delta = (1 - 2 * y[k]) * (d[k] + field[k])
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    field += S[:, k] * (1 - 2 * y[k])     # update the field before flipping
                    y[k] = 1 - y[k]
                    e += delta
                    if e < best_e - 1e-12:
                        best_y, best_e = y.copy(), e
    return best_y
