"""Search for instance families where step 2.5 matters: solve each fractional instance with the
exact step 2.5 (MILP) and with step 2.5 switched off, and compare the tree sizes.

Run:  .venv/bin/python experiments/find_hard_families.py
"""
import time

import numpy as np
from compare import print_table

from wta_bpc import bpc
from wta_bpc.instances import fractional_instances

# (weapons, targets, density)
FAMILIES = [(10, 10, 0.3), (10, 10, 0.5), (15, 15, 0.3), (15, 15, 0.5), (20, 10, 0.3),
            (20, 10, 0.5), (20, 20, 0.3), (20, 20, 0.5), (30, 15, 0.3),
            (30, 15, 0.2), (40, 20, 0.2), (40, 20, 0.3), (50, 25, 0.2), (60, 20, 0.2)]
INSTANCES_PER_FAMILY = 10


def no_heuristic(candidates, pool, weapons, targets):
    """Step 2.5 switched off: only integral LPs (no free target) give an incumbent."""
    return [] if not targets else None


def main():
    rows = []
    for n, m, density in FAMILIES:
        start = time.perf_counter()
        instances, tries = fractional_instances(n, m, density, INSTANCES_PER_FAMILY, seed=2)
        with_25 = [bpc.solve_wta(P, v)[2] for P, v in instances]
        without = [bpc.solve_wta(P, v, residual_solver=no_heuristic)[2] for P, v in instances]
        ratios = [b / a for a, b in zip(with_25, without)]
        rows.append({"family": f"{n}x{m} d={density}", "fractional": f"{len(instances)}/{tries}",
                     "nodes med": int(np.median(with_25)), "nodes max": max(with_25),
                     "without 2.5 med": int(np.median(without)), "without 2.5 max": max(without),
                     "ratio mean": round(float(np.mean(ratios)), 2), "ratio max": round(max(ratios), 2),
                     "seconds": round(time.perf_counter() - start, 1)})
        print_table(rows[-1:])                 # progress, one family at a time
    print()
    print_table(rows)
    print("\nnodes: tree size with the exact step 2.5 (MILP); without 2.5: residual_solver returns None "
          "at fractional nodes; ratio: nodes without / nodes with, per instance")


if __name__ == "__main__":
    main()
