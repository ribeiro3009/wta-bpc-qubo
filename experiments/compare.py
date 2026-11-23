"""Compare the exact MILP and the QUBO solvers inside step 2.5 of the branch-and-price.

For every instance (sparse, fractional root LP) and every solver, run the full branch-and-price
with that solver in step 2.5, and record:
  - per step-2.5 call: QUBO size (variables, couplings, one-hot states), solver time, cost of the
    answer vs the exact optimum of the same residual problem (computed apart, not timed);
    for the QAOA solvers, the exact probabilities of measuring an optimal / a feasible vector,
    and the same probabilities under uniform sampling (the baselines);
  - per run: final value, number of nodes, total time.
Writes experiments/results/calls.csv and runs.csv and prints two summary tables.

Run:  .venv/bin/python experiments/compare.py
"""
import csv
import math
import time
from pathlib import Path

import numpy as np

from wta_bpc import bpc
from wta_bpc.instances import fractional_instances
from wta_bpc.qubo import build_qubo, feasible_mask, groups_by_target, solve_residual_qubo
from wta_bpc.solvers import qaoa_xy

FAMILIES = [(8, 8, 0.5), (10, 10, 0.5), (12, 8, 0.5)]       # (weapons, targets, density)
INSTANCES_PER_FAMILY = 20
QAOA_X_MAX_QUBITS = 16                       # statevector over 2^K: larger residuals skipped
SMALL = QAOA_X_MAX_QUBITS                    # "small" calls: those where every solver runs

# name in the tables -> (solver for solve_residual_qubo, or "milp" / "no_heuristic"; options)
SOLVERS = {
    "milp": ("milp", {}),
    "no_heuristic": ("no_heuristic", {}),                     # step 2.5 switched off
    "random_sampling": ("random", {"shots": 1000, "seed": 0}),          # best of 1000 random y
    "random_onehot": ("random_onehot", {"shots": 1000, "seed": 0}),     # same, one column/target
    "annealing": ("annealing", {"num_reads": 10, "num_sweeps": 500, "seed": 0}),
    "annealing_cheap": ("annealing", {"num_reads": 3, "num_sweeps": 100, "seed": 0}),
    "qaoa_x": ("qaoa", {"reps": 2, "maxiter": 200, "shots": 1000, "seed": 0}),
    "qaoa_xy": ("qaoa_xy", {"reps": 2, "maxiter": 200, "shots": 1000, "seed": 0}),
}
QAOA_SOLVERS = ["qaoa_x", "qaoa_xy"]

RESULTS = Path(__file__).parent / "results"


def cost(cols, pool):
    return sum(pool[col] for col in cols)


def too_large(name, K, onehot_states):
    """True if this solver cannot run on a residual of this size."""
    if name == "qaoa_x":
        return K > QAOA_X_MAX_QUBITS
    if name == "qaoa_xy":
        return onehot_states > qaoa_xy.MAX_STATES
    return False


def exact_probabilities(name, info, Q, const, index, weapons, targets, best):
    """P(optimal) and P(feasible) of the final QAOA state, and the same under uniform sampling
    (over all 2^K vectors for qaoa_x, over the one-hot vectors for qaoa_xy)."""
    probs = info["probabilities"]
    if name == "qaoa_x":
        K = len(index)
        Y = (np.arange(2 ** K)[:, None] >> np.arange(K)) & 1        # same order as the statevector
        energies = np.einsum("ni,ij,nj->n", Y, Q, Y) + const
        feasible = feasible_mask(Y, index, weapons, targets)
    else:
        energies = info["energies"] + const
        feasible = qaoa_xy.subspace_conflicts(Q, groups_by_target(index, targets)) == 0
    optimal = feasible & np.isclose(energies, best)
    return {"p_optimal": probs[optimal].sum(), "p_feasible": probs[feasible].sum(),
            "p_optimal_uniform": optimal.mean(), "p_feasible_uniform": feasible.mean()}


def recording_solver(name, calls):
    """A residual_solver for solve_wta that runs solver `name` and appends one record per call."""
    solver, options = SOLVERS[name]

    def residual_solver(candidates, pool, weapons, targets):
        if not targets:                        # integral LP: nothing left to solve
            return []
        K = len(candidates)
        Q, const, index = build_qubo(candidates, pool, weapons, targets)
        record = {
            "variables": K,
            "couplings": int(np.count_nonzero(np.triu(Q, 1))),
            # vectors with one column per free target: the space the XY mixer explores
            "onehot_states": math.prod(len(group) for group in groups_by_target(index, targets)),
        }
        best = cost(bpc.solve_residual(candidates, pool, weapons, targets), pool)   # reference
        info = {}

        start = time.perf_counter()
        if solver == "milp":
            cols, status = bpc.solve_residual(candidates, pool, weapons, targets), "ok"
        elif solver == "no_heuristic" or too_large(name, K, record["onehot_states"]):
            cols, status = None, "skipped"
        else:
            extra = {"info": info} if name in QAOA_SOLVERS else {}
            cols = solve_residual_qubo(candidates, pool, weapons, targets, solver=solver,
                                       **options, **extra)
            status = "ok" if cols is not None else "infeasible"
        record["seconds"] = time.perf_counter() - start

        found = cost(cols, pool) if cols is not None else None
        record.update({"status": status, "cost": found, "optimal_cost": best,
                       "optimal": found is not None and abs(found - best) < 1e-9})
        if info:
            record.update(exact_probabilities(name, info, Q, const, index, weapons, targets, best))
        calls.append(record)
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
        fields = list(dict.fromkeys(key for row in rows for key in row))     # union of all keys
        with open(RESULTS / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summarize(call_rows, run_rows)


# ---------- summary tables ----------

def print_table(rows):
    """Print a list of dicts as a table, one column per key."""
    headers = list(rows[0])
    widths = [max(len(h), *(len(str(row[h])) for row in rows)) for h in headers]
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print("  ".join(str(row[h]).rjust(w) for h, w in zip(headers, widths)))


def fraction(count, total):
    return f"{count}/{total}"


def mean(values, digits=3):
    return round(float(np.mean(values)), digits) if values else "-"


def summarize(call_rows, run_rows):
    milp = {(r["family"], r["instance"]): r for r in run_rows if r["solver"] == "milp"}
    families = list(dict.fromkeys(r["family"] for r in run_rows))

    main, probabilities = [], []
    for family in families:
        for name in SOLVERS:
            calls = [c for c in call_rows if c["family"] == family and c["solver"] == name]
            runs = [r for r in run_rows if r["family"] == family and r["solver"] == name]
            ran = [c for c in calls if c["status"] != "skipped"]
            answered = [c for c in ran if c["cost"] is not None]
            small = [c for c in calls if c["variables"] <= SMALL]
            sizes = sorted(c["variables"] for c in calls)
            main.append({
                "family": family,
                "solver": name,
                "calls": len(calls),
                "K med": sizes[len(sizes) // 2],
                "K max": sizes[-1],
                "skipped": len(calls) - len(ran),
                "infeasible": len(ran) - len(answered),
                "optimal": fraction(sum(c["optimal"] for c in ran), len(ran)),
                f"optimal K<={SMALL}": fraction(sum(c["optimal"] for c in small), len(small)),
                "mean gap": mean([c["cost"] - c["optimal_cost"] for c in answered]),
                "ms/call": mean([1000 * c["seconds"] for c in ran], 1),
                "nodes": sum(r["nodes"] for r in runs),
                "nodes MILP": sum(milp[(family, r["instance"])]["nodes"] for r in runs),
                "exact": fraction(sum(abs(r["value"] - milp[(family, r["instance"])]["value"]) < 1e-9
                                      for r in runs), len(runs)),
            })
            if name in QAOA_SOLVERS:
                small_ran = [c for c in ran if c["variables"] <= SMALL]
                probabilities.append({
                    "family": family,
                    "solver": name,
                    "calls run": len(ran),
                    "P(opt)": mean([c["p_optimal"] for c in ran], 4),
                    "P(opt) uniform": mean([c["p_optimal_uniform"] for c in ran], 4),
                    "P(feas)": mean([c["p_feasible"] for c in ran]),
                    "P(feas) uniform": mean([c["p_feasible_uniform"] for c in ran]),
                    f"P(opt) K<={SMALL}": mean([c["p_optimal"] for c in small_ran], 4),
                })

    print()
    print_table(main)
    print("\ncalls: step-2.5 calls with a non-empty residual; K: QUBO variables; skipped: solver not run "
          f"(no_heuristic: always; qaoa_x: K > {QAOA_X_MAX_QUBITS}; qaoa_xy: more than "
          f"{qaoa_xy.MAX_STATES} one-hot states); infeasible: answer broke a rule (None returned); "
          "optimal: answer as good as the exact residual optimum; mean gap: extra cost vs that optimum; "
          "nodes: tree size; exact: final value equal to the MILP run")
    print()
    print_table(probabilities)
    print("\nP(opt), P(feas): exact probabilities of measuring an optimal / feasible vector after QAOA, "
          "mean over the calls run; uniform: the same for a uniform random vector (over 2^K vectors "
          "for qaoa_x, over the one-hot vectors for qaoa_xy)")


if __name__ == "__main__":
    run()
