"""Tests of solve_residual_qubo: drop-in replacement of bpc.solve_residual."""
import importlib.util

import numpy as np
import pytest

import wta_bpc.qubo as qubo
from wta_bpc import bpc
from wta_bpc.qubo import solve_residual_qubo

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None

# (solver name, options passed to its solve function)
SOLVERS = [
    ("bruteforce", {}),
    ("annealing", {"seed": 0}),
    pytest.param("qaoa", {"seed": 0}, marks=pytest.mark.skipif(not HAS_QISKIT, reason="no qiskit")),
    ("qaoa_xy", {"seed": 0}),
    ("random_onehot", {"seed": 0}),
]


def cost(cols, pool):
    return sum(pool[col] for col in cols)


@pytest.mark.parametrize("solver, options", SOLVERS)
def test_same_answer_as_milp_on_sparse_residual(sparse_residual, solver, options):
    candidates, pool, weapons, targets = sparse_residual
    milp_cols = bpc.solve_residual(candidates, pool, weapons, targets)
    qubo_cols = solve_residual_qubo(candidates, pool, weapons, targets, solver=solver, **options)
    assert cost(qubo_cols, pool) == pytest.approx(cost(milp_cols, pool))
    assert sorted(j for j, t in qubo_cols) == sorted(targets)          # each free target once


def test_no_free_target_gives_empty_plan():
    assert solve_residual_qubo([], {}, [0, 1], []) == []


def test_infeasible_answer_returns_none(sparse_residual, monkeypatch):
    """If the solver returns a vector that breaks a rule, we must get None, never a bad plan."""
    def bad_solver(Q, **options):
        return np.zeros(Q.shape[0], dtype=int)                         # no column for any target

    monkeypatch.setattr(qubo, "get_solver", lambda name: bad_solver)
    assert solve_residual_qubo(*sparse_residual) is None


def test_unknown_solver(sparse_residual):
    with pytest.raises(ValueError):
        solve_residual_qubo(*sparse_residual, solver="magic")


# ---------- end to end: the branch-and-price with a QUBO step 2.5 ----------

P_SPARSE = np.array([[0.0, 0.0, 0.0, 0.9],
                     [0.0, 0.6, 0.7, 0.6],
                     [0.0, 0.6, 0.9, 0.0],
                     [0.9, 0.9, 0.0, 0.0]])
V_SPARSE = np.array([4.0, 9.0, 6.0, 10.0])


@pytest.mark.parametrize("solver, options", SOLVERS)
def test_branch_and_price_with_qubo_heuristic(solver, options):
    def residual_solver(candidates, pool, weapons, targets):
        return solve_residual_qubo(candidates, pool, weapons, targets, solver=solver, **options)

    value, assign, nodes = bpc.solve_wta(P_SPARSE, V_SPARSE, residual_solver=residual_solver)
    assert value == pytest.approx(5.6)
    assert assign == [3, 1, 2, 0]
    assert nodes == 3                    # same tree as with the MILP: the incumbent was as good


def test_default_is_still_milp(monkeypatch):
    """Without residual_solver, solve_wta must keep calling bpc.solve_residual (the MILP)."""
    calls = []
    original = bpc.solve_residual
    monkeypatch.setattr(bpc, "solve_residual", lambda *args: calls.append(1) or original(*args))
    bpc.solve_wta(P_SPARSE, V_SPARSE)
    assert len(calls) > 0
