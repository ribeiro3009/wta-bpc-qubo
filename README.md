# wta-bpc-qubo

Branch-and-price for the static **Weapon-Target Assignment (WTA)** problem, with a QUBO
(quantum-ready) version of its primal heuristic.

## The problem

Weapons `i = 0..n-1`, targets `j = 0..m-1`. Target `j` has value `v[j]`; weapon `i` destroys
target `j` with probability `P[i, j]`. Each weapon fires at exactly one target. Minimize the
expected surviving value

```
sum_j v[j] * prod_{i -> j} (1 - P[i, j])
```

## The algorithm

A simplified version of the branch-and-price of Bertsimas & Paskov (2025),
*Naval Research Logistics*, doi:[10.1002/nav.22249](https://doi.org/10.1002/nav.22249):

- **Columns** `(j, t)`: target `j` is hit by exactly the set of weapons `t`, with cost
  `v[j] * prod_{i in t} (1 - P[i, j])`. The non-linearity is hidden in the column cost.
- **Master LP** (HiGHS): choose columns so that each target gets exactly one column and each
  weapon is used at most once. Columns are added by **column generation** (pricing per target).
- **Branch-and-bound** on the pairs (weapon, target) when the LP solution is fractional.
- **Primal heuristic** (section 2.5 of the paper): freeze the columns at 1 in the LP solution and
  solve the residual problem exactly over the candidate columns. This gives the incumbent.

## Goal of this project

Replace the exact MILP of the primal heuristic by a **QUBO**, solved by brute force, simulated
annealing and QAOA (simulated), keeping the same interface, and measure the effect on the
branch-and-price.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Run

```bash
.venv/bin/python bpc.py      # solves a random 20 x 20 instance
.venv/bin/python -m pytest   # runs the tests
```

With Docker:

```bash
docker build -t wta-bpc .
docker run --rm wta-bpc
```

## Layout

```
src/wta_bpc/     package (branch-and-price, QUBO, solvers)
tests/           tests on instances with known optimum
experiments/     instance generators and MILP vs QUBO comparisons
```
