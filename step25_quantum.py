"""Quantum version of step 2.5 (primal heuristic) of the branch-and-price in bpc.py.

Step 2.5 freezes the columns at 1 in the LP solution and solves the leftover ("residual")
problem over the candidate columns of the pool: each free target exactly one column, each free
weapon at most one column, minimum total cost. bpc.solve_residual does it with a MILP; here it
is written as a QUBO and solved by a QAOA with an XY mixer (simulated with numpy).
The branch-and-price stays exact: step 2.5 only proposes incumbents, it never prunes on its own.
"""
import math
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

import bpc

# QAOA settings (same as the experiments of the full version on the main branch)
DEPTH = 2              # number of QAOA layers p
MAXITER = 2000         # COBYLA: max evaluations (it converges well before)
TOL = 1e-8             # COBYLA: final trust radius
SHOTS = 1000           # measurements of the final state
SEED = 0               # fixed seed: same answer at every run
MAX_STATES = 2_000_000  # largest one-hot subspace we simulate

# How often the quantum step ran / was skipped / gave an infeasible answer (read by demo.py)
COUNTS = {"qaoa": 0, "too_large": 0, "infeasible": 0}


# ---------- a) the QUBO ----------

def build_qubo(candidates, pool, weapons, targets, A=None, B=None):
    """QUBO of the residual problem, one binary y_k per candidate column k = (j, t), cost c_k:

        E(y) = sum_k c_k y_k
             + A * sum_{free target g} (sum_{k: j(k) = g} y_k - 1)^2   each target exactly once
             + B * sum_{k < l, t(k) and t(l) share a weapon} y_k y_l    each weapon at most once

    With y_k^2 = y_k: Q[k, k] = c_k - A; Q[k, l] += 2A (same target); Q[k, l] += B (shared
    weapon); constant = A * (number of free targets). Then E(y) = y^T Q y + const, Q upper
    triangular. Returns (Q, const, index), with variable k <-> column index[k].

    Why A, B > max_k c_k is enough: the empty column (g, {}) of every free target is always a
    candidate, so any infeasible y can be repaired (add a missing empty column, drop a duplicate
    or a column sharing a weapon); each repair saves a penalty > max cost, so it lowers E.
    Hence the minimum of E is feasible and equals the MILP optimum.
    Default A = B = 1.1 * max cost: big penalties would only make the landscape harder."""
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
                Q[k, l] += 2 * A
            if t & t2:
                Q[k, l] += B
    return Q, A * len(targets), index


# ---------- b) checking an answer, grouping variables ----------

def is_feasible(y, index, weapons, targets):
    """True if the chosen columns cover each free target exactly once, each free weapon <= once."""
    chosen = [index[k] for k in np.flatnonzero(np.asarray(y) > 0.5)]
    targets_ok = all(sum(j == g for j, t in chosen) == 1 for g in targets)
    weapons_ok = all(sum(i in t for j, t in chosen) <= 1 for i in weapons)
    return targets_ok and weapons_ok


def groups_by_target(index, targets):
    """For each free target g, the variables k whose column is for g."""
    return [[k for k, (j, t) in enumerate(index) if j == g] for g in targets]


# ---------- c) QAOA with an XY mixer, simulated in the one-hot subspace ----------
# The XY mixer moves the single "1" of a group to another variable of the same group, so the rule
# "exactly one column per target" holds at all times and penalty A is never needed. The state
# then lives in the one-hot subspace: one column per target, D = prod_g |S_g| states instead of
# 2^K. We store it as an array of shape (|S_1|, ..., |S_G|): entry (a_1, ..., a_G) is the
# amplitude of "target g takes its a_g-th column".
# Simulating only this subspace is a classical trick: a real quantum machine still needs K qubits.

def subspace_energies(Q, groups):
    """y^T Q y for every one-hot vector, as an array of shape (|S_1|, ..., |S_G|).
    E(a) = sum_g Q[s_g, s_g] + sum_{g<h} (Q + Q^T)[s_g, s_h], with s_g = groups[g][a_g]."""
    G, pair = len(groups), Q + Q.T
    C = np.zeros([len(group) for group in groups])
    for g in range(G):
        shape = [1] * G
        shape[g] = len(groups[g])
        C = C + np.diag(Q)[groups[g]].reshape(shape)                        # along axis g
        for h in range(g + 1, G):
            shape = [1] * G
            shape[g], shape[h] = len(groups[g]), len(groups[h])
            C = C + pair[np.ix_(groups[g], groups[h])].reshape(shape)       # axes g and h
    return C


def qaoa_xy(Q, groups):
    """Run the XY-mixer QAOA and measure it SHOTS times; return the measured one-hot vector y
    of lowest energy. Returns None if the subspace has more than MAX_STATES states."""
    if math.prod(len(group) for group in groups) > MAX_STATES:
        return None
    C = subspace_energies(Q, groups)
    spread = C.std()                                   # normalized cost: angles of order 1
    C_norm = (C - C.mean()) / spread if spread > 0 else np.zeros_like(C)
    start_state = np.full(C.shape, 1 / np.sqrt(C.size), dtype=complex)   # uniform over subspace
    mixers = [np.ones((n, n)) - np.eye(n) for n in C.shape]   # XY mixer on one-hot = "ones - I"

    def state(x):                                      # x = (gamma_1..gamma_p, beta_1..beta_p)
        psi = start_state
        for layer in range(DEPTH):
            psi = psi * np.exp(-1j * x[layer] * C_norm)                       # cost phase
            for g, M in enumerate(mixers):                                      # mixer, group g
                U = expm(-1j * x[DEPTH + layer] * M)
                psi = np.moveaxis(np.tensordot(U, psi, axes=(1, g)), 0, g)
        return psi

    def mean_energy(x):
        return float(np.sum(np.abs(state(x)) ** 2 * C_norm))

    # Start from a "linear ramp" (gamma grows, beta shrinks, like a discretized annealing),
    # slightly perturbed; COBYLA tunes the 2p angles to minimize <H>.
    rng = np.random.default_rng(SEED)
    steps = (np.arange(DEPTH) + 0.5) / DEPTH
    x0 = np.concatenate([0.8 * steps, 0.8 * (1 - steps)]) + rng.normal(0, 0.05, 2 * DEPTH)
    result = minimize(mean_energy, x0, method="COBYLA", tol=TOL, options={"maxiter": MAXITER})
    probs = (np.abs(state(result.x)) ** 2).ravel()

    # Measure SHOTS times, keep the best measured vector.
    rng = np.random.default_rng(SEED)
    measured = np.unique(rng.choice(probs.size, size=SHOTS, p=probs / probs.sum()))
    best = measured[np.argmin(C.ravel()[measured])]
    y = np.zeros(Q.shape[0], dtype=int)
    for group, a in zip(groups, np.unravel_index(best, C.shape)):
        y[group[a]] = 1
    return y


# ---------- d) drop-in replacement for bpc.solve_residual ----------

def quantum_residual_solver(candidates, pool, weapons, targets):
    """Same interface as bpc.solve_residual: the chosen columns, or None (no incumbent from
    this node; the branch-and-price stays exact)."""
    if not targets:                                    # integral LP: nothing left to solve
        return []
    Q, const, index = build_qubo(candidates, pool, weapons, targets)
    y = qaoa_xy(Q, groups_by_target(index, targets))
    if y is None:
        COUNTS["too_large"] += 1
        return None
    COUNTS["qaoa"] += 1
    if not is_feasible(y, index, weapons, targets):    # always check the solver's answer
        COUNTS["infeasible"] += 1
        return None
    return [index[k] for k in np.flatnonzero(y)]


# ---------- e) single entry point ----------

def solve(P, v, step25="qaoa_xy"):
    """Solve the WTA instance (P[i, j] kill probability of weapon i on target j, v[j] values).
    step25: "qaoa_xy" (quantum step 2.5) or "milp" (Pierre's original MILP).
    Returns (optimal value, assignment weapon -> target, number of branch-and-bound nodes)."""
    if step25 not in ("qaoa_xy", "milp"):
        raise ValueError(f"step25 must be 'qaoa_xy' or 'milp', not {step25!r}")
    solver = quantum_residual_solver if step25 == "qaoa_xy" else bpc.solve_residual
    return bpc.solve_wta(np.asarray(P, float), np.asarray(v, float), residual_solver=solver)
