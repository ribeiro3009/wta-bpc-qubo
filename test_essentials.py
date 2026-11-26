"""Two quick checks of the quantum step 2.5. Run from this folder:  python -m pytest test_essentials.py"""
import itertools
import numpy as np

import bpc
from demo import fractional_instances
from step25_quantum import build_qubo, is_feasible, solve


def residual_problems(count):
    """Real residual problems: the arguments of every bpc.solve_residual call made while
    solving `count` fractional instances (with a non-empty set of free targets)."""
    problems = []

    def recording_solver(candidates, pool, weapons, targets):
        if targets:
            problems.append((candidates, dict(pool), weapons, targets))
        return bpc.solve_residual(candidates, pool, weapons, targets)

    for P, v in fractional_instances(count):
        bpc.solve_wta(P, v, residual_solver=recording_solver)
    return problems


def test_qubo_minimum_is_feasible_and_equals_milp():
    """Brute force over all 2^K vectors: the QUBO minimum is feasible and costs the MILP optimum."""
    checked = 0
    for candidates, pool, weapons, targets in residual_problems(3):
        Q, const, index = build_qubo(candidates, pool, weapons, targets)
        K = len(index)
        if K > 18:                                     # keep the brute force fast
            continue
        Y = np.array(list(itertools.product([0, 1], repeat=K)))
        energies = np.einsum("ni,ij,nj->n", Y, Q, Y) + const
        y = Y[np.argmin(energies)]
        milp_cost = sum(pool[col] for col in bpc.solve_residual(candidates, pool, weapons, targets))
        assert is_feasible(y, index, weapons, targets)
        assert abs(energies.min() - milp_cost) < 1e-9
        checked += 1
    assert checked > 0


def test_qaoa_xy_gives_same_value_as_milp():
    for P, v in fractional_instances(3):
        value_milp, _, _ = solve(P, v, step25="milp")
        value_qaoa, assign, _ = solve(P, v, step25="qaoa_xy")
        assert abs(value_milp - value_qaoa) < 1e-9
        assert abs(bpc.wta_value(assign, P, v) - value_qaoa) < 1e-9
