"""Tests of the XY-mixer QAOA simulated in the one-hot subspace (numpy only, no qiskit needed)."""
import itertools

import numpy as np
import pytest

from wta_bpc import bpc
from wta_bpc.qubo import build_qubo, energy, groups_by_target, is_feasible, solve_residual_qubo
from wta_bpc.solvers import qaoa_xy


def one_hot_vectors(groups, K):
    """Every vector with one variable per group, with its position a = (a_1, ..., a_G)."""
    for a in itertools.product(*[range(len(group)) for group in groups]):
        y = np.zeros(K, dtype=int)
        for group, a_g in zip(groups, a):
            y[group[a_g]] = 1
        yield a, y


@pytest.mark.parametrize("case", ["hand_case", "sparse_residual"])
def test_subspace_energies_and_conflicts(case, request):
    candidates, pool, weapons, targets = request.getfixturevalue(case)
    Q, const, index = build_qubo(candidates, pool, weapons, targets)
    groups = groups_by_target(index, targets)
    C = qaoa_xy.subspace_energies(Q, groups)
    conflicts = qaoa_xy.subspace_conflicts(Q, groups)
    assert C.shape == tuple(len(group) for group in groups)
    for a, y in one_hot_vectors(groups, len(index)):
        assert C[a] == pytest.approx(y @ Q @ y)                       # same energy as the QUBO
        assert (conflicts[a] == 0) == is_feasible(y, index, weapons, targets)


def test_xy_beats_uniform_on_hand_case(hand_case):
    """Hand case: 9 one-hot states, 1 optimal (uniform 1/9 = 11%). XY QAOA p=1 gives about 25%."""
    Q, const, index = build_qubo(*hand_case, A=10, B=10)
    groups = groups_by_target(index, hand_case[3])
    probs, C = qaoa_xy.qaoa_xy_probabilities(Q, groups, reps=1, seed=0)
    assert probs.sum() == pytest.approx(1)
    optimum = np.unravel_index(np.argmin(C), C.shape)
    assert probs[optimum] > 2 / C.size


def test_more_layers_help_on_sparse_residual(sparse_residual):
    """4x4 residual: 100 one-hot states, 1 optimal. P(optimum) grows with the depth p."""
    candidates, pool, weapons, targets = sparse_residual
    Q, const, index = build_qubo(candidates, pool, weapons, targets)
    groups = groups_by_target(index, targets)
    conflicts = qaoa_xy.subspace_conflicts(Q, groups)
    p_opt = []
    for reps in [1, 3]:
        probs, C = qaoa_xy.qaoa_xy_probabilities(Q, groups, reps=reps, seed=0)
        optimal = (conflicts == 0) & np.isclose(C, C[conflicts == 0].min())
        p_opt.append(probs[optimal].sum())
    assert p_opt[0] > 1 / 100                                          # better than uniform
    assert p_opt[1] > p_opt[0]


def test_solve_hand_case(hand_case):
    Q, const, index = build_qubo(*hand_case, A=10, B=10)
    y = qaoa_xy.solve(Q, groups_by_target(index, hand_case[3]), seed=0)
    assert energy(Q, const, y) == pytest.approx(4.2)


def test_solve_residual_matches_milp(sparse_residual):
    candidates, pool, weapons, targets = sparse_residual
    milp = sum(pool[c] for c in bpc.solve_residual(candidates, pool, weapons, targets))
    cols = solve_residual_qubo(candidates, pool, weapons, targets, solver="qaoa_xy", seed=0)
    assert sum(pool[c] for c in cols) == pytest.approx(milp)


def test_too_large_subspace_is_refused():
    groups = [list(range(10 * g, 10 * g + 10)) for g in range(7)]      # 10^7 states
    with pytest.raises(ValueError):
        qaoa_xy.solve(np.zeros((70, 70)), groups)
