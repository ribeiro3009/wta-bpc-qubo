"""QAOA with an XY mixer per free target, simulated exactly in the one-hot subspace (numpy only).

Groups: S_g = the variables (candidate columns) of free target g. The XY mixer keeps the rule
"exactly one column per target" true at all times, so the state lives in the subspace of vectors
with one variable per group: dimension D = prod_g |S_g| instead of 2^K. A state is stored as an
array of shape (|S_1|, ..., |S_G|): entry (a_1, ..., a_G) is the amplitude of "target g takes its
a_g-th column".

- Cost: on one-hot vectors the penalty A vanishes, and y^T Q y + const = sum of column costs
  + B * (pairs of chosen columns sharing a weapon). So we reuse Q from build_qubo:
  E(a) = sum_g Q[s_g, s_g] + sum_{g<h} (Q + Q^T)[s_g, s_h],  with s_g = S_g[a_g].
- Initial state: uniform superposition in each group (a W state), i.e. uniform over the subspace.
- Mixer: restricted to one-hot vectors, sum_{k<l in S_g} (X_k X_l + Y_k Y_l) / 2 is the matrix
  "all ones minus identity" (complete graph on S_g): it moves the "1" of group g to another
  variable. exp(-i beta M_g) is applied to axis g of the array, group by group.

On quantum hardware this still needs K qubits: simulating only the subspace is a classical trick.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

MAX_STATES = 2_000_000


def _sum_over_subspace(diag, pair, groups):
    """Array of shape (|S_1|, ..., |S_G|) whose entry a is
    sum_g diag[s_g] + sum_{g<h} pair[s_g, s_h], with s_g = groups[g][a_g]."""
    G = len(groups)
    total = np.zeros([len(group) for group in groups])
    for g in range(G):
        shape = [1] * G
        shape[g] = len(groups[g])
        total = total + diag[groups[g]].reshape(shape)              # broadcast along axis g
        for h in range(g + 1, G):
            shape = [1] * G
            shape[g], shape[h] = len(groups[g]), len(groups[h])
            total = total + pair[np.ix_(groups[g], groups[h])].reshape(shape)   # axes g and h
    return total


def subspace_energies(Q, groups):
    """y^T Q y for every one-hot vector, as an array of shape (|S_1|, ..., |S_G|)."""
    return _sum_over_subspace(np.diag(Q), Q + Q.T, groups)


def subspace_conflicts(Q, groups):
    """Number of pairs of chosen columns sharing a weapon, for every one-hot vector.
    Uses the structure of build_qubo: between two different targets, Q is B > 0 exactly when
    the two columns share a weapon, and 0 otherwise."""
    return _sum_over_subspace(np.zeros(Q.shape[0]), (Q + Q.T > 0).astype(float), groups)


def qaoa_xy_probabilities(Q, groups, reps=1, maxiter=200, seed=None):
    """Run the XY-mixer QAOA (depth reps). Returns (probabilities, energies), two arrays of shape
    (|S_1|, ..., |S_G|): probability of each one-hot vector after QAOA, and its y^T Q y."""
    C = subspace_energies(Q, groups)
    if C.size > MAX_STATES:
        raise ValueError(f"one-hot subspace of {C.size} states is too large; max {MAX_STATES}")
    # Normalized cost: zero mean and unit standard deviation over the subspace (angles of order 1).
    spread = C.std()
    C_norm = (C - C.mean()) / spread if spread > 0 else np.zeros_like(C)
    start_state = np.full(C.shape, 1 / np.sqrt(C.size), dtype=complex)
    mixers = [np.ones((n, n)) - np.eye(n) for n in C.shape]

    def state(x):
        psi = start_state
        for layer in range(reps):
            psi = psi * np.exp(-1j * x[layer] * C_norm)                      # cost phase
            for g, M in enumerate(mixers):                                     # XY mixer, group g
                U = expm(-1j * x[reps + layer] * M)
                psi = np.moveaxis(np.tensordot(U, psi, axes=(1, g)), 0, g)
        return psi

    def mean_energy(x):
        return float(np.sum(np.abs(state(x)) ** 2 * C_norm))

    rng = np.random.default_rng(seed)
    steps = (np.arange(reps) + 0.5) / reps
    x0 = np.concatenate([0.8 * steps, 0.8 * (1 - steps)]) + rng.normal(0, 0.05, 2 * reps)
    result = minimize(mean_energy, x0, method="COBYLA", options={"maxiter": maxiter})
    return np.abs(state(result.x)) ** 2, C


def solve(Q, groups, reps=1, maxiter=200, shots=1000, seed=None, info=None):
    """Return the measured one-hot vector of lowest energy y^T Q y after the XY-mixer QAOA.
    info: optional dict; receives "probabilities" and "energies" (arrays over the subspace)."""
    K = Q.shape[0]
    if not groups:
        return np.zeros(K, dtype=int)
    probs, C = qaoa_xy_probabilities(Q, groups, reps=reps, maxiter=maxiter, seed=seed)
    if info is not None:
        info["probabilities"], info["energies"] = probs, C
    rng = np.random.default_rng(seed)
    flat = probs.ravel()
    measured = np.unique(rng.choice(flat.size, size=shots, p=flat / flat.sum()))
    best = measured[np.argmin(C.ravel()[measured])]
    choice = np.unravel_index(best, C.shape)                   # a_g for every group g
    y = np.zeros(K, dtype=int)
    for group, a in zip(groups, choice):
        y[group[a]] = 1
    return y
