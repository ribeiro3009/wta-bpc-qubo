"""Tests of the QUBO of the residual problem (step 2.5)."""
import itertools

import numpy as np
import pytest

from wta_bpc import bpc
from wta_bpc.qubo import build_qubo, energy, is_feasible

# Case 3 (hand-computed): free targets 1 (v = 9) and 2 (v = 6), free weapons 1 and 2.
F = frozenset
HAND_POOL = {
    (1, F({1})): 3.6,
    (1, F({1, 2})): 1.44,
    (1, F()): 9.0,
    (2, F({2})): 0.6,
    (2, F({1})): 1.8,
    (2, F()): 6.0,
}
HAND_CANDIDATES = list(HAND_POOL)
HAND_WEAPONS, HAND_TARGETS = [1, 2], [1, 2]


def all_vectors(K):
    """Every 0/1 vector of length K, one per row (2^K rows)."""
    return np.array(list(itertools.product([0, 1], repeat=K)))


def test_hand_qubo_matrix():
    Q, const, index = build_qubo(HAND_CANDIDATES, HAND_POOL, HAND_WEAPONS, HAND_TARGETS, A=10, B=10)
    assert index == HAND_CANDIDATES
    assert const == 20                                   # A * 2 free targets
    assert np.allclose(np.diag(Q), [3.6 - 10, 1.44 - 10, -1, 0.6 - 10, 1.8 - 10, -4])
    assert Q[0, 1] == 2 * 10 + 10                        # same target AND share weapon 1
    assert Q[0, 2] == 2 * 10                             # same target only
    assert Q[0, 4] == 10                                 # share weapon 1 only
    assert Q[0, 3] == 0                                  # different targets, no common weapon
    assert np.allclose(np.tril(Q, -1), 0)                # upper triangular


def test_hand_qubo_minimum():
    Q, const, index = build_qubo(HAND_CANDIDATES, HAND_POOL, HAND_WEAPONS, HAND_TARGETS, A=10, B=10)
    energies = {tuple(y): energy(Q, const, y) for y in all_vectors(len(index))}
    best = min(energies, key=energies.get)
    assert energies[best] == pytest.approx(4.2)
    assert [index[k] for k in np.flatnonzero(best)] == [(1, F({1})), (2, F({2}))]
    infeasible = [e for y, e in energies.items() if not is_feasible(y, index, HAND_WEAPONS, HAND_TARGETS)]
    assert min(infeasible) == pytest.approx(10.6)


def test_energy_equals_cost_when_feasible():
    """On feasible vectors the penalties vanish: E(y) is the total cost of the chosen columns."""
    Q, const, index = build_qubo(HAND_CANDIDATES, HAND_POOL, HAND_WEAPONS, HAND_TARGETS, A=10, B=10)
    for y in all_vectors(len(index)):
        if is_feasible(y, index, HAND_WEAPONS, HAND_TARGETS):
            cost = sum(HAND_POOL[index[k]] for k in np.flatnonzero(y))
            assert energy(Q, const, y) == pytest.approx(cost)


def test_is_feasible():
    # (1,{1}) + (2,{2}): feasible
    assert is_feasible([1, 0, 0, 1, 0, 0], HAND_CANDIDATES, HAND_WEAPONS, HAND_TARGETS)
    # target 2 has no column
    assert not is_feasible([1, 0, 0, 0, 0, 0], HAND_CANDIDATES, HAND_WEAPONS, HAND_TARGETS)
    # target 1 has two columns
    assert not is_feasible([1, 0, 1, 1, 0, 0], HAND_CANDIDATES, HAND_WEAPONS, HAND_TARGETS)
    # weapon 1 used twice: (1,{1}) + (2,{1})
    assert not is_feasible([1, 0, 0, 0, 1, 0], HAND_CANDIDATES, HAND_WEAPONS, HAND_TARGETS)


def test_sparse_4x4_residual_qubo_matches_milp(monkeypatch):
    """On the real step-2.5 problem of the sparse 4x4 instance (14 candidates), the minimum of
    the QUBO with default penalties is feasible and equals the MILP optimum."""
    P = np.array([[0.0, 0.0, 0.0, 0.9],
                  [0.0, 0.6, 0.7, 0.6],
                  [0.0, 0.6, 0.9, 0.0],
                  [0.9, 0.9, 0.0, 0.0]])
    v = np.array([4.0, 9.0, 6.0, 10.0])

    # Run the root of the branch-and-price and record the arguments given to solve_residual.
    calls = []
    original = bpc.solve_residual

    def spy(candidates, pool, weapons, targets):
        calls.append((candidates, dict(pool), weapons, targets))
        return original(candidates, pool, weapons, targets)

    monkeypatch.setattr(bpc, "solve_residual", spy)
    bpc.solve_wta(P, v)
    candidates, pool, weapons, targets = calls[0]          # the call made at the root node
    assert len(candidates) == 14

    milp_cols = original(candidates, pool, weapons, targets)
    milp_value = sum(pool[col] for col in milp_cols)

    Q, const, index = build_qubo(candidates, pool, weapons, targets)
    Y = all_vectors(len(index))                            # 2^14 = 16384 vectors
    energies = np.einsum("ni,ij,nj->n", Y, Q, Y) + const   # E(y) for every row y at once
    best = Y[np.argmin(energies)]
    assert is_feasible(best, index, weapons, targets)
    assert energies.min() == pytest.approx(milp_value)
