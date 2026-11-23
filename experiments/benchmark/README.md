# Step-2.5 benchmark

Real QUBO-ready problems produced by a branch-and-price for static Weapon-Target Assignment.
Each one is the **residual problem** of the primal heuristic (section 2.5 of Bertsimas & Paskov
2025): after freezing the columns at 1 in the LP solution, choose among the remaining candidate
columns so that

- every free target gets **exactly one** column,
- every free weapon is used **at most once**,
- the total cost is minimal.

A column `(j, t)` means "target `j` is struck by exactly the weapons in `t`". Its cost is
`v[j] * prod_{i in t} (1 - P[i, j])`, the expected surviving value of target `j`. The empty
column `(j, {})`, with cost `v[j]`, is always a candidate, so every problem has a feasible solution.

## Files

One JSON file per instance family, `residuals_{n}x{m}_d{density}.json`. A family is n weapons,
m targets, and kill probabilities that are non-zero with probability `density`. Each file is a
list of problems:

| field | meaning |
|---|---|
| `name` | `{n}x{m}_d{density}_inst{i}_call{k}`: instance `i` of the family, `k`-th distinct residual problem met in its tree |
| `weapons` | free weapons (indices in the original instance) |
| `targets` | free targets (indices in the original instance) |
| `columns` | candidate columns: `{"target": j, "weapons": [...], "cost": c}` |
| `optimum` | exact optimal cost (MILP, scipy / HiGHS) |
| `variables` | K = number of columns = number of QUBO variables (qubits with the standard encoding) |
| `onehot_states` | D = product over free targets of their number of columns: size of the "one column per target" space explored by an XY-mixer QAOA |

Only the residual problems are stored, not the original instances.

| file | problems | K |
|---|---|---|
| `residuals_8x8_d0.5.json` | 25 | 11 to 35 |
| `residuals_10x10_d0.5.json` | 22 | 11 to 36 |
| `residuals_12x8_d0.5.json` | 29 | 14 to 58 |

## Use

```python
from wta_bpc import benchmark
from wta_bpc.qubo import build_qubo

problems = benchmark.load("experiments/benchmark/residuals_8x8_d0.5.json")
candidates, pool, weapons, targets = benchmark.problem_from_dict(problems[0])
Q, const, index = build_qubo(candidates, pool, weapons, targets)   # E(y) = y^T Q y + const
```

The penalties of `build_qubo` default to 1.1 x the largest column cost. With those, the minimum
of E is feasible and equals `optimum`.

## Regenerate

The instances come from a fixed seed, so the files are reproducible:

```bash
.venv/bin/python experiments/export_benchmark.py
```

The families and the number of instances per family (20) are set in `experiments/compare.py`.
