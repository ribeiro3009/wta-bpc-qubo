# WTA branch-and-price with a quantum step 2.5 (essential version)

## The idea

- `bpc.py` solves the static Weapon-Target Assignment exactly by branch-and-price (Bertsimas & Paskov 2025). At each node, **step 2.5** (primal heuristic) freezes the columns at 1 in the LP solution and solves a small *residual* problem: pick candidate columns so that each free target gets exactly one and each free weapon is used at most once, at minimum cost.
- `step25_quantum.py` writes that residual problem as a **QUBO**: one binary variable per candidate column, plus penalties A (one column per target) and B (no shared weapon). With A, B > the largest cost, the QUBO minimum is the residual optimum.
- The QUBO is solved by **QAOA with an XY mixer** per target. The XY mixer keeps "one column per target" true at all times, so penalty A is never needed. It is simulated with numpy, only in the one-column-per-target subspace (a classical trick: real hardware needs one qubit per candidate column).
- The answer is always checked. A QAOA answer that breaks a rule is dropped. Step 2.5 only proposes incumbents and never prunes on its own, so **the final result stays exact** whatever the QAOA finds.

## Install

Python 3.10+.

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python demo.py                          # MILP vs QAOA step 2.5 on 10 sparse 8x8 instances 
python -m pytest test_essentials.py     # 2 tests, a few seconds
```

## Use it from another program (e.g. the GUI)

```python
from step25_quantum import solve

value, assignment, nodes = solve(P, v)  # step 2.5 by QAOA XY (default)
value, assignment, nodes = solve(P, v, step25="milp")  # original MILP step 2.5
```

- `P[i][j]`: probability that weapon `i` destroys target `j` (n weapons x m targets).
- `v[j]`: value of target `j`.
- `value`: expected surviving value (minimized). `assignment[i]`: target of weapon `i`. `nodes`: branch-and-bound nodes.

Note: the GUI currently asks only for `P`, not for the target values `v`. Until it does, it must pass some `v` (for example all ones).

## Files

| File | Content |
|---|---|
| `bpc.py` | Branch-and-price (unchanged copy of `src/wta_bpc/bpc.py` on main) |
| `step25_quantum.py` | QUBO, XY-mixer QAOA, drop-in residual solver, `solve(P, v)` |
| `demo.py` | Comparison MILP vs QAOA on sparse instances with a fractional root LP |
| `test_essentials.py` | QUBO minimum = MILP optimum; same final value with both step 2.5 |

The full version (other QUBO solvers, QAOA with the X mixer, experiments, benchmark, Docker, full test suite) is on the **main** branch.
