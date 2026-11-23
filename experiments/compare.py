"""Compare the exact MILP and the QUBO solvers inside step 2.5 of the branch-and-price.

For every instance (sparse, fractional root LP) and every solver, run the full branch-and-price
with that solver in step 2.5, and record:
  - per step-2.5 call: QUBO size (variables, couplings), solver time, cost of the answer
    vs the exact optimum of the same residual problem (computed apart, not timed);
  - per run: final value, number of nodes, total time.
Writes experiments/results/calls.csv and runs.csv and prints a summary.

Run:  .venv/bin/python experiments/compare.py
"""
import csv
import time
from pathlib import Path

import numpy as np

from wta_bpc import bpc
from wta_bpc.instances import fractional_instances
from wta_bpc.qubo import build_qubo, solve_residual_qubo

FAMILIES = [(8, 8, 0.5), (10, 10, 0.5), (12, 8, 0.5)]       # (weapons, targets, density)
INSTANCES_PER_FAMILY = 20
QAOA_MAX_QUBITS = 16                                          # larger residuals: QAOA skipped

# name -> (QUBO solver or None for the MILP, options)
SOLVERS = {
    "milp": (None, {}),
    "annealing": ("annealing", {"num_reads": 10, "num_sweeps": 500, "seed": 0}),
    "annealing_cheap": ("annealing", {"num_reads": 3, "num_sweeps": 100, "seed": 0}),
    "qaoa": ("qaoa", {"reps": 1, "shots": 1000, "seed": 0}),
}

RESULTS = Path(__file__).parent / "results"


def cost(cols, pool):
    return sum(pool[col] for col in cols)


def recording_solver(name, calls):
    """A residual_solver for solve_wta that runs solver `name` and appends one record per call."""
    qubo_solver, options = SOLVERS[name]

    def residual_solver(candidates, pool, weapons, targets):
        if not targets:                        # integral LP: nothing left to solve
            return []
        K = len(candidates)
        Q, _, _ = build_qubo(candidates, pool, weapons, targets)
        couplings = int(np.count_nonzero(np.triu(Q, 1)))
        best = cost(bpc.solve_residual(candidates, pool, weapons, targets), pool)   # reference

        start = time.perf_counter()
        if qubo_solver is None:
            cols, status = bpc.solve_residual(candidates, pool, weapons, targets), "ok"
        elif qubo_solver == "qaoa" and K > QAOA_MAX_QUBITS:
            cols, status = None, "skipped"
        else:
            cols = solve_residual_qubo(candidates, pool, weapons, targets, solver=qubo_solver, **options)
            status = "ok" if cols is not None else "infeasible"
        seconds = time.perf_counter() - start

        found = cost(cols, pool) if cols is not None else None
        calls.append({"variables": K, "couplings": couplings, "seconds": seconds, "status": status,
                      "cost": found, "optimal_cost": best,
                      "optimal": found is not None and abs(found - best) < 1e-9})
        return cols

    return residual_solver


def run():
    RESULTS.mkdir(exist_ok=True)
    call_rows, run_rows = [], []
    for n, m, density in FAMILIES:
        family = f"{n}x{m} d={density}"
        instances, tries = fractional_instances(n, m, density, INSTANCES_PER_FAMILY, seed=1)
        print(f"{family}: {len(instances)} fractional instances out of {tries} generated")
        for number, (P, v) in enumerate(instances):
            for name in SOLVERS:
                calls = []
                start = time.perf_counter()
                value, _, nodes = bpc.solve_wta(P, v, residual_solver=recording_solver(name, calls))
                seconds = time.perf_counter() - start
                run_rows.append({"family": family, "instance": number, "solver": name,
                                 "value": value, "nodes": nodes, "seconds": seconds})
                call_rows += [{"family": family, "instance": number, "solver": name, **c} for c in calls]

    for filename, rows in [("calls.csv", call_rows), ("runs.csv", run_rows)]:
        with open(RESULTS / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summarize(call_rows, run_rows)


def summarize(call_rows, run_rows):
    milp_value = {(r["family"], r["instance"]): r["value"] for r in run_rows if r["solver"] == "milp"}
    milp_nodes = {(r["family"], r["instance"]): r["nodes"] for r in run_rows if r["solver"] == "milp"}
    print()
    print(f"{'family':<14}{'solver':<17}{'calls':>6}{'K med':>6}{'K max':>6}{'skip':>6}{'infeas':>7}"
          f"{'optimal':>9}{'mean gap':>10}{'ms/call':>9}{'nodes':>7}{'MILP':>6}{'exact':>7}")
    for family in dict.fromkeys(r["family"] for r in run_rows):
        for name in SOLVERS:
            calls = [c for c in call_rows if c["family"] == family and c["solver"] == name]
            runs = [r for r in run_rows if r["family"] == family and r["solver"] == name]
            solved = [c for c in calls if c["status"] != "skipped"]
            answered = [c for c in solved if c["cost"] is not None]
            sizes = sorted(c["variables"] for c in calls)
            gap = np.mean([c["cost"] - c["optimal_cost"] for c in answered]) if answered else float("nan")
            ms = 1000 * np.mean([c["seconds"] for c in solved]) if solved else float("nan")
            exact = sum(abs(r["value"] - milp_value[(family, r["instance"])]) < 1e-9 for r in runs)
            print(f"{family:<14}{name:<17}{len(calls):>6}{sizes[len(sizes) // 2]:>6}{sizes[-1]:>6}"
                  f"{len(calls) - len(solved):>6}{len(solved) - len(answered):>7}"
                  f"{sum(c['optimal'] for c in solved):>5}/{len(solved):<3}{gap:>10.3f}{ms:>9.1f}"
                  f"{sum(r['nodes'] for r in runs):>7}"
                  f"{sum(milp_nodes[(family, r['instance'])] for r in runs):>6}{exact:>4}/{len(runs)}")
    print("\ncalls: step-2.5 calls with a non-empty residual; K: QUBO variables; skip: QAOA not run (K > "
          f"{QAOA_MAX_QUBITS}); infeas: solver answer broke a rule (None returned); optimal: answer as good "
          "as the exact residual optimum; mean gap: extra cost vs that optimum; nodes vs MILP: tree size; "
          "exact: final value equal to the MILP run")


if __name__ == "__main__":
    run()
