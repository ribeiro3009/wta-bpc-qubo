"""Demo: solve sparse WTA instances with step 2.5 done by the MILP and by the XY-mixer QAOA.

Only instances whose root LP is fractional are kept: on the others step 2.5 has nothing to do.
Run from this folder:  python demo.py
"""

import time

import numpy as np

import bpc
import step25_quantum
from step25_quantum import solve

WEAPONS, TARGETS, DENSITY, INSTANCES = 8, 8, 0.5, 10


def sparse_instance(rng):
    """Each pair (i, j) is effective with probability DENSITY, then P[i, j] ~ U(0.1, 0.9),
    else 0. Target values are integers in 1..10."""
    P = rng.uniform(0.1, 0.9, (WEAPONS, TARGETS)) * (
        rng.random((WEAPONS, TARGETS)) < DENSITY
    )
    return P, rng.integers(1, 11, TARGETS).astype(float)


def root_is_fractional(P, v):
    """Solve the root LP exactly as bpc.solve_wta does; True if its solution is fractional."""
    master = bpc.Master(WEAPONS, TARGETS)
    for j, t in dict.fromkeys(bpc.initial_columns(P, v)):
        master.add(j, t, bpc.state_cost(j, t, P, v))
    _, z = bpc.column_generation(master, P, v, frozenset())
    return bpc.branch_pair(master.pool, z, WEAPONS, TARGETS) is not None


def fractional_instances(count, seed=1):
    rng = np.random.default_rng(seed)
    found = []
    while len(found) < count:
        P, v = sparse_instance(rng)
        if root_is_fractional(P, v):
            found.append((P, v))
    return found


if __name__ == "__main__":
    print(
        f"{INSTANCES} sparse {WEAPONS}x{TARGETS} instances (density {DENSITY}), fractional root LP\n"
    )
    print(
        " inst |  MILP value  QAOA value  same | nodes MILP  nodes QAOA | QAOA runs | time MILP  time QAOA"
    )
    for n, (P, v) in enumerate(fractional_instances(INSTANCES)):
        start = time.perf_counter()
        value_milp, _, nodes_milp = solve(P, v, step25="milp")
        time_milp = time.perf_counter() - start

        step25_quantum.COUNTS.update(qaoa=0, too_large=0, infeasible=0)
        start = time.perf_counter()
        value_qaoa, _, nodes_qaoa = solve(P, v, step25="qaoa_xy")
        time_qaoa = time.perf_counter() - start

        same = "yes" if abs(value_milp - value_qaoa) < 1e-9 else "NO"
        counts = step25_quantum.COUNTS
        extra = (
            f"  ({counts['too_large']} too large, {counts['infeasible']} infeasible)"
            if counts["too_large"] or counts["infeasible"]
            else ""
        )
        print(
            f" {n:4d} | {value_milp:11.4f} {value_qaoa:11.4f}  {same:>4} | {nodes_milp:10d} {nodes_qaoa:11d} "
            f"| {counts['qaoa']:9d} | {time_milp:8.2f}s {time_qaoa:9.2f}s{extra}"
        )
