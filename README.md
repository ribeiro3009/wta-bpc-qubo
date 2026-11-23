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

## Getting started

### 1. Just run the tests (Docker)

To check that everything works, without installing Python.
Prerequisite: [Docker](https://docs.docker.com/get-docker/) (on Windows: Docker Desktop with WSL 2).

```bash
docker build -t wta-bpc .
docker run --rm wta-bpc            # runs the tests
```

The image includes Qiskit, so the experiments run too. To keep their results on your computer,
mount `experiments/results` as a volume:

```bash
# macOS / Linux
docker run --rm -v "$(pwd)/experiments/results:/app/experiments/results" wta-bpc python experiments/compare.py
```

```powershell
# Windows PowerShell
docker run --rm -v "${PWD}/experiments/results:/app/experiments/results" wta-bpc python experiments/compare.py
```

```bat
:: Windows cmd
docker run --rm -v "%cd%\experiments\results:/app/experiments/results" wta-bpc python experiments/compare.py
```

The CSV files then appear in `experiments/results/`. Running times are about 1.4x those of a
local install (`compare.py`: about 9 min). An optional last argument sets the output folder,
e.g. `... python experiments/compare.py experiments/results/my_run`.

### 2. Full local install

To work on the code and run the experiments. Requires **Python >= 3.11**: check with
`python3 --version` (on macOS the default `python3` may be 3.9; install a newer one from
[python.org](https://www.python.org/downloads/) or call it by version, e.g. `python3.13`).

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,quantum]"
.venv/bin/python -m pytest
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,quantum]"
.venv\Scripts\python -m pytest
```

Alternative with [uv](https://docs.astral.sh/uv/), which also downloads Python 3.13 if needed:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev,quantum]"
```

The `quantum` extra installs Qiskit. Two experiments need it: `compare.py` runs the X-mixer
QAOA (`qaoa` solver) and `benchmark_qaoa.py` imports it. The branch-and-price, the classical
solvers and the XY-mixer QAOA (`qaoa_xy`, numpy only) work without it. To skip Qiskit, install
`-e ".[dev]"`: the Qiskit tests are then skipped, and the two scripts above will fail.

### Experiments

Run from the repository root with `.venv/bin/python` (Windows: `.venv\Scripts\python`).
Results go to `experiments/results/`. Times measured on a recent laptop with a local install.

| command | what it does | time |
|---|---|---|
| `python -m wta_bpc.bpc` | solves a random 20 x 20 instance (demo) | < 1 s |
| `python experiments/compare.py` | branch-and-price with each solver in step 2.5 (MILP, annealing, QAOA X / XY, random baselines): quality, time, tree size | ~6.5 min |
| `python experiments/export_benchmark.py` | saves the step-2.5 problems to `experiments/benchmark/` | ~5 s |
| `python experiments/benchmark_qaoa.py` | QAOA X vs XY on the benchmark, p = 1..3: P(optimal), P(feasible), approximation ratio | ~9.5 min |
| `python experiments/find_hard_families.py` | tree size with / without step 2.5 on larger families | ~20 s |

## QUBO solvers for step 2.5

| name | method |
|---|---|
| `bruteforce` | enumerates all 2^K vectors (validation, K <= 22) |
| `annealing` | simulated annealing (classical reference) |
| `qaoa` | QAOA with X mixer and penalties, Qiskit statevector (K <= 20) |
| `qaoa_xy` | QAOA with an XY mixer per target, simulated in the one-column-per-target subspace (numpy). On hardware it still needs K qubits. |
| `random`, `random_onehot` | baselines: best of `shots` uniform random vectors (over all vectors / one column per target) |

## Benchmark

`experiments/benchmark/` holds the residual problems of step 2.5 met by the branch-and-price on
sparse instances with a fractional root LP (candidate columns, costs, free weapons and targets,
exact optimum). Load them with `wta_bpc.benchmark.load` and `problem_from_dict`.
See [experiments/benchmark/README.md](experiments/benchmark/README.md) for the format.

## Layout

```
src/wta_bpc/            package: branch-and-price, QUBO, solvers, instance generator, benchmark I/O
tests/                  tests on instances with known optimum
experiments/            experiment scripts (results go to experiments/results/, not versioned)
experiments/benchmark/  the step-2.5 benchmark (JSON)
```
