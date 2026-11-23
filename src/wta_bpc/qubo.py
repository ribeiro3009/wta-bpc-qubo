"""QUBO version of the residual problem of the primal heuristic (step 2.5).

One binary variable y_k per candidate column k = (j, t), with cost c_k:

    E(y) = sum_k c_k y_k
         + A * sum_{free target g} (sum_{k: j(k) = g} y_k - 1)^2     each free target exactly once
         + B * sum_{k < l, t(k) and t(l) share a weapon} y_k y_l      each weapon at most once

Expanding the square with y_k^2 = y_k gives
    diagonal Q[k, k] = c_k - A,
    Q[k, l] += 2A if k and l are columns of the same target,
    Q[k, l] += B  if k and l share a weapon,
    constant = A * (number of free targets),
so that E(y) = y^T Q y + constant, with Q upper triangular.

Choice of A, B: the empty column (g, {}) of every free target is always a candidate, costs v_g,
and no column of target g costs more. If A, B > max_k c_k, any infeasible y can be repaired
(add a missing empty column, drop a duplicate or a column sharing a weapon) into a feasible
y with lower energy, so the minimum of E is feasible and equals the residual MILP optimum.
"""
import numpy as np


def build_qubo(candidates, pool, weapons, targets, A=None, B=None):
    """Build the QUBO of the residual problem.

    candidates: list of columns (j, t), t a frozenset of weapons; pool: {column: cost}.
    weapons, targets: the free weapons and targets.
    A, B: penalty weights (default 1.1 * the largest candidate cost).
    Returns Q (K x K, upper triangular), the constant, and index = list of columns
    (variable k <-> column index[k])."""
    index = list(candidates)
    K = len(index)
    costs = np.array([pool[col] for col in index], dtype=float)
    default = 1.1 * costs.max() if K > 0 else 1.0
    A = default if A is None else A
    B = default if B is None else B

    Q = np.zeros((K, K))
    for k, (j, t) in enumerate(index):
        Q[k, k] = costs[k] - A
        for l in range(k + 1, K):
            j2, t2 = index[l]
            if j == j2:
                Q[k, l] += 2 * A          # two columns for the same target
            if t & t2:
                Q[k, l] += B              # two columns using the same weapon
    const = A * len(targets)
    return Q, const, index


def energy(Q, const, y):
    """E(y) = y^T Q y + const for a 0/1 vector y."""
    y = np.asarray(y, dtype=float)
    return float(y @ Q @ y + const)


def is_feasible(y, index, weapons, targets):
    """True if the chosen columns give each free target exactly one column
    and use each free weapon at most once."""
    chosen = [index[k] for k in np.flatnonzero(np.asarray(y) > 0.5)]
    targets_ok = all(sum(j == g for j, t in chosen) == 1 for g in targets)
    weapons_ok = all(sum(i in t for j, t in chosen) <= 1 for i in weapons)
    return targets_ok and weapons_ok


def feasible_mask(Y, index, weapons, targets):
    """Vectorized is_feasible: Y has one 0/1 vector per row; returns one bool per row.
    T[g, k] = 1 if column k is for target g; W[i, k] = 1 if column k uses weapon i.
    Y @ T.T counts the columns of each target, Y @ W.T the uses of each weapon."""
    T = np.array([[j == g for j, t in index] for g in targets], dtype=int).reshape(len(targets), len(index))
    W = np.array([[i in t for j, t in index] for i in weapons], dtype=int).reshape(len(weapons), len(index))
    Y = np.asarray(Y)
    return np.all(Y @ T.T == 1, axis=1) & np.all(Y @ W.T <= 1, axis=1)


def groups_by_target(index, targets):
    """For each free target g, the list of variables k whose column is for g."""
    return [[k for k, (j, t) in enumerate(index) if j == g] for g in targets]


# Solvers that also need the groups (variables of each free target): solve(Q, groups, ...)
GROUP_SOLVERS = {"random_onehot", "qaoa_xy"}


def get_solver(name):
    """The solve function of a solver: "bruteforce", "annealing", "qaoa", "random" take (Q, ...);
    "random_onehot" and "qaoa_xy" take (Q, groups, ...).
    QAOA is imported only when asked for, since qiskit is optional."""
    if name == "bruteforce":
        from wta_bpc.solvers import bruteforce
        return bruteforce.solve
    if name == "annealing":
        from wta_bpc.solvers import annealing
        return annealing.solve
    if name == "qaoa":
        from wta_bpc.solvers import qaoa
        return qaoa.solve
    if name == "random":
        from wta_bpc.solvers import random_sampling
        return random_sampling.solve
    if name == "random_onehot":
        from wta_bpc.solvers import random_sampling
        return random_sampling.solve_onehot
    if name == "qaoa_xy":
        from wta_bpc.solvers import qaoa_xy
        return qaoa_xy.solve
    raise ValueError(f"unknown solver {name!r}")


def solve_residual_qubo(candidates, pool, weapons, targets, solver="annealing", A=None, B=None,
                        **solver_options):
    """Same interface as bpc.solve_residual, through a QUBO: build it, solve it, check the answer.
    Returns the chosen columns, or None if the solver's answer breaks a rule
    (the branch-and-price stays exact: it just gets no incumbent from this node)."""
    if not targets:
        return []
    Q, const, index = build_qubo(candidates, pool, weapons, targets, A=A, B=B)
    if solver in GROUP_SOLVERS:
        y = get_solver(solver)(Q, groups_by_target(index, targets), **solver_options)
    else:
        y = get_solver(solver)(Q, **solver_options)
    if not is_feasible(y, index, weapons, targets):
        return None
    return [index[k] for k in np.flatnonzero(np.asarray(y) > 0.5)]
