"""Export the distinct non-empty step-2.5 problems met by the branch-and-price (MILP version) on
the sparse fractional instances of compare.py, one JSON file per family, in experiments/benchmark/.

Run:  .venv/bin/python experiments/export_benchmark.py
"""
from pathlib import Path

from compare import FAMILIES, INSTANCES_PER_FAMILY

from wta_bpc import bpc, benchmark
from wta_bpc.instances import fractional_instances

OUT = Path(__file__).parent / "benchmark"


def residual_problems(P, v):
    """The distinct non-empty residual problems step 2.5 receives while solving (P, v)."""
    seen = {}

    def recording(candidates, pool, weapons, targets):
        cols = bpc.solve_residual(candidates, pool, weapons, targets)
        if targets:
            key = (tuple(candidates), tuple(weapons), tuple(targets))
            seen.setdefault(key, (list(candidates), dict(pool), list(weapons), list(targets),
                                  sum(pool[c] for c in cols)))
        return cols

    bpc.solve_wta(P, v, residual_solver=recording)
    return list(seen.values())


def main():
    OUT.mkdir(exist_ok=True)
    for n, m, density in FAMILIES:
        instances, _ = fractional_instances(n, m, density, INSTANCES_PER_FAMILY, seed=1)
        problems = []
        for number, (P, v) in enumerate(instances):
            for k, (candidates, pool, weapons, targets, optimum) in enumerate(residual_problems(P, v)):
                name = f"{n}x{m}_d{density}_inst{number}_call{k}"
                problems.append(benchmark.problem_to_dict(name, candidates, pool, weapons, targets, optimum))
        path = OUT / f"residuals_{n}x{m}_d{density}.json"
        benchmark.save(problems, path)
        sizes = sorted(p["variables"] for p in problems)
        print(f"{path.name}: {len(problems)} problems, K from {sizes[0]} to {sizes[-1]}")


if __name__ == "__main__":
    main()
