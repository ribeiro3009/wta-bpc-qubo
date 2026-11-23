"""Tests of the QAOA solver (skipped if qiskit, the 'quantum' extra, is not installed)."""
import numpy as np
import pytest

pytest.importorskip("qiskit")

from wta_bpc.qubo import build_qubo, energy, is_feasible  # noqa: E402
from wta_bpc.solvers import qaoa  # noqa: E402


def all_vectors(K):
    """Row i = the vector y with y_k = bit k of i (the qubit order of the statevector)."""
    return (np.arange(2 ** K)[:, None] >> np.arange(K)) & 1


def test_hamiltonian_diagonal_is_qubo_energy(hand_case):
    """On every basis state, <H> must be a * y^T Q y + b with a > 0: same minimizer as the QUBO."""
    Q, _, _ = build_qubo(*hand_case, A=10, B=10)
    Y = all_vectors(Q.shape[0])
    E = np.einsum("ni,ij,nj->n", Y, Q, Y)
    diag = np.real(np.diag(qaoa.qubo_to_hamiltonian(Q).to_matrix()))
    a, b = np.polyfit(E, diag, 1)
    assert a > 0
    assert np.allclose(a * E + b, diag)


def test_qaoa_hand_case(hand_case):
    Q, const, index = build_qubo(*hand_case, A=10, B=10)
    y = qaoa.solve(Q, seed=0)
    assert energy(Q, const, y) == pytest.approx(4.2)


def test_qaoa_concentrates_on_optimum(hand_case):
    """After optimizing the angles, the optimum is more likely than under uniform guessing."""
    Q, _, _ = build_qubo(*hand_case, A=10, B=10)
    Y = all_vectors(Q.shape[0])
    optimum = np.argmin(np.einsum("ni,ij,nj->n", Y, Q, Y))
    probs = qaoa.qaoa_probabilities(Q, reps=1, seed=0)
    assert probs.sum() == pytest.approx(1)
    assert probs[optimum] > 2 / len(probs)               # at least twice the uniform 1/64


def test_qaoa_sparse_residual_favors_feasible(sparse_residual):
    """14 qubits: only 27 of the 16384 vectors are feasible (0.16%); QAOA must raise that a lot."""
    candidates, pool, weapons, targets = sparse_residual
    Q, _, index = build_qubo(candidates, pool, weapons, targets)
    probs = qaoa.qaoa_probabilities(Q, reps=1, seed=0)
    feasible = np.array([is_feasible(y, index, weapons, targets) for y in all_vectors(len(index))])
    assert feasible.sum() == 27
    assert probs[feasible].sum() > 10 * feasible.mean()


def test_qaoa_limits():
    assert len(qaoa.solve(np.zeros((0, 0)))) == 0
    with pytest.raises(ValueError):
        qaoa.solve(np.zeros((25, 25)))
