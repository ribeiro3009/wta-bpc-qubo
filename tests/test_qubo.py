"""Tests of the QUBO of the residual problem (step 2.5)."""
import itertools

import numpy as np
import pytest

from wta_bpc import bpc
from wta_bpc.qubo import build_qubo, energy, feasible_mask, groups_by_target, is_feasible

F = frozenset


def all_vectors(K):
    """Every 0/1 vector of length K, one per row (2^K rows)."""
    return np.array(list(itertools.product([0, 1], repeat=K)))


def test_hand_qubo_matrix(hand_case):
    Q, const, index = build_qubo(*hand_case, A=10, B=10)
    assert index == hand_case[0]
    assert const == 20                                   # A * 2 free targets
    assert np.allclose(np.diag(Q), [3.6 - 10, 1.44 - 10, -1, 0.6 - 10, 1.8 - 10, -4])
    assert Q[0, 1] == 2 * 10 + 10                        # same target AND share weapon 1
    assert Q[0, 2] == 2 * 10                             # same target only
    assert Q[0, 4] == 10                                 # share weapon 1 only
    assert Q[0, 3] == 0                                  # different targets, no common weapon
    assert np.allclose(np.tril(Q, -1), 0)                # upper triangular


def test_hand_qubo_minimum(hand_case):
    candidates, pool, weapons, targets = hand_case
    Q, const, index = build_qubo(*hand_case, A=10, B=10)
    energies = {tuple(y): energy(Q, const, y) for y in all_vectors(len(index))}
    best = min(energies, key=energies.get)
    assert energies[best] == pytest.approx(4.2)
    assert [index[k] for k in np.flatnonzero(best)] == [(1, F({1})), (2, F({2}))]
    infeasible = [e for y, e in energies.items() if not is_feasible(y, index, weapons, targets)]
    assert min(infeasible) == pytest.approx(10.6)


def test_energy_equals_cost_when_feasible(hand_case):
    """On feasible vectors the penalties vanish: E(y) is the total cost of the chosen columns."""
    candidates, pool, weapons, targets = hand_case
    Q, const, index = build_qubo(*hand_case, A=10, B=10)
    for y in all_vectors(len(index)):
        if is_feasible(y, index, weapons, targets):
            cost = sum(pool[index[k]] for k in np.flatnonzero(y))
            assert energy(Q, const, y) == pytest.approx(cost)


def test_is_feasible(hand_case):
    candidates, _, weapons, targets = hand_case
    # (1,{1}) + (2,{2}): feasible
    assert is_feasible([1, 0, 0, 1, 0, 0], candidates, weapons, targets)
    # target 2 has no column
    assert not is_feasible([1, 0, 0, 0, 0, 0], candidates, weapons, targets)
    # target 1 has two columns
    assert not is_feasible([1, 0, 1, 1, 0, 0], candidates, weapons, targets)
    # weapon 1 used twice: (1,{1}) + (2,{1})
    assert not is_feasible([1, 0, 0, 0, 1, 0], candidates, weapons, targets)


def test_feasible_mask_agrees_with_is_feasible(hand_case, sparse_residual):
    for candidates, _, weapons, targets in [hand_case, sparse_residual]:
        Y = all_vectors(len(candidates))
        expected = [is_feasible(y, candidates, weapons, targets) for y in Y]
        assert list(feasible_mask(Y, candidates, weapons, targets)) == expected


def test_groups_by_target(hand_case):
    candidates, _, _, targets = hand_case
    assert groups_by_target(candidates, targets) == [[0, 1, 2], [3, 4, 5]]


def test_sparse_4x4_residual_qubo_matches_milp(sparse_residual):
    """On the real step-2.5 problem of the sparse 4x4 instance (14 candidates), the minimum of
    the QUBO with default penalties is feasible and equals the MILP optimum."""
    candidates, pool, weapons, targets = sparse_residual
    assert len(candidates) == 14

    milp_cols = bpc.solve_residual(candidates, pool, weapons, targets)
    milp_value = sum(pool[col] for col in milp_cols)

    Q, const, index = build_qubo(candidates, pool, weapons, targets)
    Y = all_vectors(len(index))                            # 2^14 = 16384 vectors
    energies = np.einsum("ni,ij,nj->n", Y, Q, Y) + const   # E(y) for every row y at once
    best = Y[np.argmin(energies)]
    assert is_feasible(best, index, weapons, targets)
    assert energies.min() == pytest.approx(milp_value)
