"""Tests of the branch-and-price on instances with a known optimum."""
import itertools

import numpy as np
import pytest

from wta_bpc import bpc

# Case 1: mini-WTA, 3 weapons x 2 targets. The root LP is already integral.
P_MINI = np.array([[0.5, 0.2],
                   [0.4, 0.6],
                   [0.3, 0.5]])
V_MINI = np.array([10.0, 6.0])

# Case 2: sparse 4 x 4 instance whose root LP is fractional.
P_SPARSE = np.array([[0.0, 0.0, 0.0, 0.9],
                     [0.0, 0.6, 0.7, 0.6],
                     [0.0, 0.6, 0.9, 0.0],
                     [0.9, 0.9, 0.0, 0.0]])
V_SPARSE = np.array([4.0, 9.0, 6.0, 10.0])


def brute_force(P, v):
    """Try every assignment (m^n of them): only for tiny instances."""
    n, m = P.shape
    best = min(itertools.product(range(m), repeat=n), key=lambda a: bpc.wta_value(a, P, v))
    return bpc.wta_value(best, P, v), list(best)


def root_lp(P, v):
    """Build the master LP as solve_wta does and run column generation at the root."""
    n, m = P.shape
    master = bpc.Master(n, m)
    for j, t in dict.fromkeys(bpc.initial_columns(P, v)):
        master.add(j, t, bpc.state_cost(j, t, P, v))
    lp, z = bpc.column_generation(master, P, v, frozenset())
    return master, lp, z


def test_mini_wta():
    value, assign, nodes = bpc.solve_wta(P_MINI, V_MINI)
    assert value == pytest.approx(5.9)
    assert assign == [0, 1, 0]          # weapons 0 and 2 -> target 0, weapon 1 -> target 1
    assert nodes == 1                   # root LP integral: no branching


def test_sparse_4x4():
    value, assign, nodes = bpc.solve_wta(P_SPARSE, V_SPARSE)
    assert value == pytest.approx(5.6)
    assert assign == [3, 1, 2, 0]
    assert nodes == 3


def test_sparse_4x4_root_and_heuristic(monkeypatch):
    """Root LP is fractional (5.57); step 2.5 sees 14 candidate columns and finds the optimum."""
    master, lp, z = root_lp(P_SPARSE, V_SPARSE)
    assert lp == pytest.approx(5.57)
    assert np.any((z > bpc.EPS) & (z < 1 - bpc.EPS))       # some column is fractional

    # Wrap solve_residual to record how many candidate columns it receives.
    seen = []
    original = bpc.solve_residual

    def spy(candidates, pool, weapons, targets):
        seen.append(len(candidates))
        return original(candidates, pool, weapons, targets)

    monkeypatch.setattr(bpc, "solve_residual", spy)
    plan = bpc.primal_heuristic(master.pool, z, *P_SPARSE.shape)
    assert seen == [14]
    assert sum(master.pool[col] for col in plan) == pytest.approx(5.6)


@pytest.mark.parametrize("P, v", [(P_MINI, V_MINI), (P_SPARSE, V_SPARSE)])
def test_known_cases_match_brute_force(P, v):
    value, assign, _ = bpc.solve_wta(P, v)
    bf_value, _ = brute_force(P, v)
    assert value == pytest.approx(bf_value)


@pytest.mark.parametrize("seed", range(5))
def test_small_random_match_brute_force(seed):
    P, v = bpc.random_instance(4, 3, seed=seed)
    value, _, _ = bpc.solve_wta(P, v)
    bf_value, _ = brute_force(P, v)
    assert value == pytest.approx(bf_value)
