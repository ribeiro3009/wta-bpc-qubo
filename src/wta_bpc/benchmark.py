"""Benchmark of real step-2.5 problems: save / load residual problems met by the branch-and-price.

A problem is stored as a JSON object:
    {"name", "weapons": [free weapons], "targets": [free targets],
     "columns": [{"target": j, "weapons": [...], "cost": c}, ...],      (the candidate columns)
     "optimum": exact optimal cost (MILP), "variables": K, "onehot_states": D}
"""
import json
import math


def problem_to_dict(name, candidates, pool, weapons, targets, optimum):
    return {
        "name": name,
        "weapons": [int(i) for i in weapons],
        "targets": [int(j) for j in targets],
        "columns": [{"target": int(j), "weapons": sorted(int(i) for i in t), "cost": float(pool[(j, t)])}
                    for j, t in candidates],
        "optimum": float(optimum),
        "variables": len(candidates),
        "onehot_states": int(math.prod(sum(j == g for j, t in candidates) for g in targets)),
    }


def problem_from_dict(d):
    """Back to the arguments of solve_residual: (candidates, pool, weapons, targets)."""
    candidates = [(c["target"], frozenset(c["weapons"])) for c in d["columns"]]
    pool = {(c["target"], frozenset(c["weapons"])): c["cost"] for c in d["columns"]}
    return candidates, pool, d["weapons"], d["targets"]


def save(problems, path):
    with open(path, "w") as f:
        json.dump(problems, f, indent=1)


def load(path):
    """List of problem dicts; use problem_from_dict(p) to get solve_residual's arguments."""
    with open(path) as f:
        return json.load(f)
