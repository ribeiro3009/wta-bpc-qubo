"""Instances where step 2.5 has work to do: sparse P and a fractional root LP.

On uniform random instances (bpc.random_instance) the root LP is almost always integral, so the
primal heuristic only freezes columns. Even sparse instances give a fractional root LP only
2-10% of the time, so we generate many and keep the fractional ones (rejection sampling).
"""
import numpy as np

from wta_bpc import bpc


def sparse_instance(n, m, density, rng):
    """n weapons, m targets; each pair (i, j) is effective with probability `density`,
    then P[i, j] ~ U(0.1, 0.9), else P[i, j] = 0. Target values are integers in 1..10."""
    mask = rng.random((n, m)) < density
    P = rng.uniform(0.1, 0.9, (n, m)) * mask
    v = rng.integers(1, 11, m).astype(float)
    return P, v


def root_is_fractional(P, v):
    """Solve the root LP as solve_wta does; True if its solution is fractional."""
    n, m = P.shape
    master = bpc.Master(n, m)
    for j, t in dict.fromkeys(bpc.initial_columns(P, v)):
        master.add(j, t, bpc.state_cost(j, t, P, v))
    _, z = bpc.column_generation(master, P, v, frozenset())
    return bpc.branch_pair(master.pool, z, n, m) is not None


def fractional_instances(n, m, density, count, seed=0, max_tries=100_000):
    """`count` sparse instances with a fractional root LP. Returns (instances, number tried)."""
    rng = np.random.default_rng(seed)
    found = []
    for tries in range(1, max_tries + 1):
        P, v = sparse_instance(n, m, density, rng)
        if root_is_fractional(P, v):
            found.append((P, v))
            if len(found) == count:
                return found, tries
    raise RuntimeError(f"only {len(found)} fractional instances in {max_tries} tries")
