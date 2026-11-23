"""Exact QUBO solver by enumeration of all 2^K vectors (validation only, K <= ~20)."""
import numpy as np

MAX_VARS = 22
CHUNK = 2 ** 16


def solve(Q):
    """Return the 0/1 vector y minimizing y^T Q y (first one found in case of ties)."""
    K = Q.shape[0]
    if K == 0:
        return np.zeros(0, dtype=int)
    if K > MAX_VARS:
        raise ValueError(f"brute force on {K} variables (2^{K} vectors) is too slow; max {MAX_VARS}")
    bits = np.arange(K)
    best_y, best_e = None, np.inf
    for start in range(0, 2 ** K, CHUNK):
        numbers = np.arange(start, min(start + CHUNK, 2 ** K))
        Y = (numbers[:, None] >> bits) & 1               # row r = binary digits of numbers[r]
        E = np.einsum("ni,ij,nj->n", Y, Q, Y)            # y^T Q y for every row at once
        r = np.argmin(E)
        if E[r] < best_e:
            best_y, best_e = Y[r].copy(), E[r]
    return best_y
