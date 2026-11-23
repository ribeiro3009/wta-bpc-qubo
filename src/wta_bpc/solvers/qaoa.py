"""QAOA for QUBO, simulated exactly with Qiskit statevectors (requires the 'quantum' extra).

1. QUBO -> cost Hamiltonian: with y_k = (1 - Z_k) / 2 (qubit k in |1> <-> column k chosen),
   y^T Q y becomes a sum of Z_k and Z_k Z_l terms plus a constant (dropped).
2. QAOA circuit of depth p: start in |+...+> (all y at once), then p times
   exp(-i gamma H) (cost) followed by exp(-i beta sum_k X_k) (standard X mixer).
3. A classical optimizer (COBYLA) tunes gamma, beta to minimize <H>.
4. Measure the final state `shots` times, return the measured y of lowest QUBO energy.
"""
import numpy as np
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

MAX_QUBITS = 20


def qubo_to_hamiltonian(Q):
    """SparsePauliOp H such that, on the basis state of y, <H> = y^T Q y - constant,
    divided by its largest coefficient (same minimizer, angles of order 1)."""
    K = Q.shape[0]
    terms = []
    for k in range(K):
        # linear term d_k y_k = d_k (1 - Z_k) / 2  ->  -d_k / 2 Z_k
        terms.append(("Z", [k], -Q[k, k] / 2))
        for l in range(k + 1, K):
            c = Q[k, l] + Q[l, k]
            if c != 0:
                # c y_k y_l = c (1 - Z_k - Z_l + Z_k Z_l) / 4
                terms.append(("ZZ", [k, l], c / 4))
                terms.append(("Z", [k], -c / 4))
                terms.append(("Z", [l], -c / 4))
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=K).simplify()
    scale = np.abs(H.coeffs).max()
    return H / scale if scale > 0 else H


def qaoa_probabilities(Q, reps=2, maxiter=200, seed=None):
    """Run QAOA (depth reps) on the QUBO Q. Returns the probability of every basis state:
    probs[i] is the probability of y with y_k = bit k of the integer i."""
    K = Q.shape[0]
    if K > MAX_QUBITS:
        raise ValueError(f"QAOA simulation on {K} qubits is too slow; max {MAX_QUBITS}")
    H = qubo_to_hamiltonian(Q)
    circuit = qaoa_ansatz(H, reps=reps)
    gammas = [p for p in circuit.parameters if p.vector.name == "γ"]
    betas = [p for p in circuit.parameters if p.vector.name == "β"]

    def state(x):
        values = dict(zip(gammas, x[:reps])) | dict(zip(betas, x[reps:]))
        return Statevector(circuit.assign_parameters(values))

    def mean_energy(x):
        return state(x).expectation_value(H).real

    # Start from a "linear ramp" (gamma grows, beta shrinks), like a discretized annealing,
    # slightly perturbed so that different seeds explore different starts.
    rng = np.random.default_rng(seed)
    steps = (np.arange(reps) + 0.5) / reps
    x0 = np.concatenate([0.8 * steps, 0.8 * (1 - steps)]) + rng.normal(0, 0.05, 2 * reps)
    result = minimize(mean_energy, x0, method="COBYLA", options={"maxiter": maxiter})
    return state(result.x).probabilities()


def solve(Q, reps=2, maxiter=200, shots=1000, seed=None):
    """Return the measured 0/1 vector of lowest energy y^T Q y after QAOA."""
    K = Q.shape[0]
    if K == 0:
        return np.zeros(0, dtype=int)
    probs = qaoa_probabilities(Q, reps=reps, maxiter=maxiter, seed=seed)
    rng = np.random.default_rng(seed)
    measured = np.unique(rng.choice(len(probs), size=shots, p=probs / probs.sum()))
    Y = (measured[:, None] >> np.arange(K)) & 1          # one measured y per row
    energies = np.einsum("ni,ij,nj->n", Y, Q, Y)
    return Y[np.argmin(energies)]
