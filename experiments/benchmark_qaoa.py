"""QAOA on the exported step-2.5 benchmark (run export_benchmark.py first).

For every problem and depth p = 1, 2, 3: exact P(optimal), P(feasible) and approximation ratio
(mean cost of the feasible outcomes / optimum) of the final QAOA state, for the X mixer
(only if K <= 16) and the XY mixer, next to the same numbers for uniform sampling
(over all 2^K vectors for X, over the one-hot vectors for XY).
Writes experiments/results/benchmark_qaoa.csv and prints a summary.

Run:  .venv/bin/python experiments/benchmark_qaoa.py
"""
import csv
from pathlib import Path

import numpy as np
from compare import print_table

from wta_bpc import benchmark
from wta_bpc.qubo import build_qubo, feasible_mask, groups_by_target
from wta_bpc.solvers import qaoa, qaoa_xy

BENCHMARK = Path(__file__).parent / "benchmark"
RESULTS = Path(__file__).parent / "results"
DEPTHS = [1, 2, 3]
X_MAX_QUBITS = 16


def stats(probs, energies, feasible, optimum):
    """P(optimal), P(feasible), approximation ratio for a distribution over vectors."""
    p_feas = probs[feasible].sum()
    p_opt = probs[feasible & np.isclose(energies, optimum)].sum()
    ratio = (probs[feasible] * energies[feasible]).sum() / p_feas / optimum if p_feas > 0 else np.nan
    return p_opt, p_feas, ratio


def all_x_distributions(Q, const, index, weapons, targets):
    """Energies and feasibility of all 2^K vectors (statevector order)."""
    K = len(index)
    Y = (np.arange(2 ** K)[:, None] >> np.arange(K)) & 1
    return np.einsum("ni,ij,nj->n", Y, Q, Y) + const, feasible_mask(Y, index, weapons, targets)


def measure(problem):
    """One row per (mixer, depth) for this problem."""
    candidates, pool, weapons, targets = benchmark.problem_from_dict(problem)
    Q, const, index = build_qubo(candidates, pool, weapons, targets)
    groups = groups_by_target(index, targets)
    optimum, K = problem["optimum"], problem["variables"]
    rows = []

    def add(mixer, depth, probs, energies, feasible):
        p_opt, p_feas, ratio = stats(probs.ravel(), energies.ravel(), feasible.ravel(), optimum)
        uniform = np.full(energies.size, 1 / energies.size)
        u_opt, u_feas, u_ratio = stats(uniform, energies.ravel(), feasible.ravel(), optimum)
        rows.append({"name": problem["name"], "family": problem["name"].split("_inst")[0],
                     "mixer": mixer, "depth": depth, "variables": K,
                     "onehot_states": problem["onehot_states"],
                     "p_optimal": p_opt, "p_feasible": p_feas, "ratio": ratio,
                     "p_optimal_uniform": u_opt, "p_feasible_uniform": u_feas, "ratio_uniform": u_ratio})

    C = qaoa_xy.subspace_energies(Q, groups) + const
    xy_feasible = qaoa_xy.subspace_conflicts(Q, groups) == 0
    if K <= X_MAX_QUBITS:
        x_energies, x_feasible = all_x_distributions(Q, const, index, weapons, targets)
    for depth in DEPTHS:
        probs, _ = qaoa_xy.qaoa_xy_probabilities(Q, groups, reps=depth, seed=0)
        add("XY", depth, probs, C, xy_feasible)
        if K <= X_MAX_QUBITS:
            add("X", depth, qaoa.qaoa_probabilities(Q, reps=depth, seed=0), x_energies, x_feasible)
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    problems = [p for path in sorted(BENCHMARK.glob("*.json")) for p in benchmark.load(path)]
    print(f"{len(problems)} benchmark problems")
    rows = [row for problem in problems for row in measure(problem)]
    with open(RESULTS / "benchmark_qaoa.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for mixer in ["X", "XY"]:
        for depth in DEPTHS:
            group = [r for r in rows if r["mixer"] == mixer and r["depth"] == depth]
            # same problems for both mixers: those small enough for X
            small = [r for r in group if r["variables"] <= X_MAX_QUBITS]
            summary.append({
                "mixer": mixer, "p": depth, "problems": len(group),
                "P(opt)": round(np.mean([r["p_optimal"] for r in group]), 4),
                "P(opt) uniform": round(np.mean([r["p_optimal_uniform"] for r in group]), 4),
                "gain med": round(np.median([r["p_optimal"] / r["p_optimal_uniform"] for r in group]), 1),
                "P(feas)": round(np.mean([r["p_feasible"] for r in group]), 3),
                "P(feas) uniform": round(np.mean([r["p_feasible_uniform"] for r in group]), 3),
                "ratio": round(np.nanmean([r["ratio"] for r in group]), 3),
                "ratio uniform": round(np.nanmean([r["ratio_uniform"] for r in group]), 3),
                f"P(opt) K<={X_MAX_QUBITS}": round(np.mean([r["p_optimal"] for r in small]), 4),
            })
    print_table(summary)
    print("\nP(opt), P(feas): exact probabilities after QAOA (mean over problems); gain med: median of "
          "P(opt) / P(opt) uniform; ratio: mean cost of the feasible outcomes / optimum (1 = always "
          f"optimal); K<={X_MAX_QUBITS}: only the problems where the X mixer also runs")


if __name__ == "__main__":
    main()
