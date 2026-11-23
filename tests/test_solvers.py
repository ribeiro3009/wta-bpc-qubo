"""Tests of the QUBO solvers: brute force (exact) and simulated annealing (heuristic)."""
import numpy as np
import pytest

from wta_bpc import bpc
from wta_bpc.qubo import build_qubo, energy, is_feasible
from wta_bpc.solvers import annealing, bruteforce

F = frozenset


def random_qubo(K, seed):
    """Random upper-triangular Q with entries in [-1, 1]."""
    rng = np.random.default_rng(seed)
    return np.triu(rng.uniform(-1, 1, (K, K)))


# ---------- brute force ----------

def test_bruteforce_hand_case(hand_case):
    Q, const, index = build_qubo(*hand_case, A=10, B=10)
    y = bruteforce.solve(Q)
    assert energy(Q, const, y) == pytest.approx(4.2)
    assert [index[k] for k in np.flatnonzero(y)] == [(1, F({1})), (2, F({2}))]


def test_bruteforce_sparse_residual(sparse_residual):
    candidates, pool, weapons, targets = sparse_residual
    Q, const, index = build_qubo(candidates, pool, weapons, targets)
    y = bruteforce.solve(Q)
    milp_value = sum(pool[col] for col in bpc.solve_residual(candidates, pool, weapons, targets))
    assert is_feasible(y, index, weapons, targets)
    assert energy(Q, const, y) == pytest.approx(milp_value)


def test_bruteforce_several_chunks():
    """K = 18 gives 2^18 vectors = 4 chunks: the minimum must not depend on the chunking."""
    Q = random_qubo(18, seed=0)
    y = bruteforce.solve(Q)
    rng = np.random.default_rng(1)
    others = rng.integers(0, 2, (2000, 18))
    assert all(energy(Q, 0, y) <= energy(Q, 0, z) + 1e-12 for z in others)


def test_bruteforce_refuses_large():
    with pytest.raises(ValueError):
        bruteforce.solve(np.zeros((30, 30)))


# ---------- simulated annealing ----------

def test_annealing_hand_case(hand_case):
    Q, const, index = build_qubo(*hand_case, A=10, B=10)
    y = annealing.solve(Q, seed=0)
    assert energy(Q, const, y) == pytest.approx(4.2)


def test_annealing_sparse_residual(sparse_residual):
    candidates, pool, weapons, targets = sparse_residual
    Q, const, index = build_qubo(candidates, pool, weapons, targets)
    y = annealing.solve(Q, seed=0)
    milp_value = sum(pool[col] for col in bpc.solve_residual(candidates, pool, weapons, targets))
    assert is_feasible(y, index, weapons, targets)
    assert energy(Q, const, y) == pytest.approx(milp_value)


@pytest.mark.parametrize("seed", range(5))
def test_annealing_matches_bruteforce_on_random_qubos(seed):
    Q = random_qubo(12, seed)
    y_exact = bruteforce.solve(Q)
    y_sa = annealing.solve(Q, seed=seed)
    assert energy(Q, 0, y_sa) == pytest.approx(energy(Q, 0, y_exact))


def test_annealing_symmetric_q_same_energy():
    """A symmetric Q (same y^T Q y as its upper-triangular version) must work too."""
    U = random_qubo(10, seed=3)
    Q_sym = (U + U.T) / 2 + np.diag(np.diag(U)) / 2      # same quadratic form as U
    y = annealing.solve(Q_sym, seed=0)
    assert energy(U, 0, y) == pytest.approx(energy(U, 0, bruteforce.solve(U)))


def test_empty_qubo():
    assert len(bruteforce.solve(np.zeros((0, 0)))) == 0
    assert len(annealing.solve(np.zeros((0, 0)))) == 0


# ---------- random sampling (baseline) ----------

def test_random_sampling_is_best_of_its_samples():
    from wta_bpc.solvers import random_sampling
    Q = random_qubo(10, seed=4)
    y = random_sampling.solve(Q, shots=50, seed=0)
    samples = np.random.default_rng(0).integers(0, 2, (50, 10))         # same draws as the solver
    assert energy(Q, 0, y) == pytest.approx(min(energy(Q, 0, z) for z in samples))


def test_random_onehot_samples_one_column_per_target(sparse_residual):
    """Even the best sample of a single shot has exactly one column per free target."""
    from wta_bpc.qubo import groups_by_target
    from wta_bpc.solvers import random_sampling
    candidates, pool, weapons, targets = sparse_residual
    Q, _, index = build_qubo(candidates, pool, weapons, targets)
    groups = groups_by_target(index, targets)
    for seed in range(20):
        y = random_sampling.solve_onehot(Q, groups, shots=1, seed=seed)
        assert [sum(y[k] for k in group) for group in groups] == [1] * len(groups)


def test_random_onehot_finds_sparse_optimum(sparse_residual):
    """The one-hot space of the 4x4 residual is small (product of group sizes), so 1000 shots
    find the optimum, while uniform bits over 2^14 vectors are mostly infeasible."""
    from wta_bpc.qubo import solve_residual_qubo
    candidates, pool, weapons, targets = sparse_residual
    milp = sum(pool[c] for c in bpc.solve_residual(candidates, pool, weapons, targets))
    cols = solve_residual_qubo(candidates, pool, weapons, targets, solver="random_onehot", seed=0)
    assert sum(pool[c] for c in cols) == pytest.approx(milp)


def test_random_sampling_with_enough_shots_finds_hand_optimum(hand_case):
    """64 states and 1000 shots: the optimum is drawn almost surely."""
    from wta_bpc.solvers import random_sampling
    Q, const, _ = build_qubo(*hand_case, A=10, B=10)
    assert energy(Q, const, random_sampling.solve(Q, seed=0)) == pytest.approx(4.2)
