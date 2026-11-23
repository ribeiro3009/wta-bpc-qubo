"""Branch-and-price for the static Weapon-Target Assignment (after Bertsimas & Paskov 2025).

Problem:  min sum_j v_j prod_i (1 - P_ij)^{x_ij},  each weapon i assigned to exactly one target.

State reformulation: a column (j, t) is a target j with the set t of weapons striking it,
cost q = v_j prod_{i in t} (1 - P_ij). Master problem = set partitioning:
every weapon in exactly one chosen column, every target in exactly one chosen column.

Simplifications vs the paper: greedy initial columns (not the convex relaxation),
no clique cuts, no aggregation of identical weapons, depth-first tree search.
"""
import math
import numpy as np
import highspy
from scipy.optimize import milp, LinearConstraint, Bounds

EPS = 1e-6


# ---------- instance, costs ----------

def random_instance(n, m, seed=0):
    """n weapons, m targets: kill probabilities P (n x m), target values v (m)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, size=(n, m)), rng.integers(1, 101, size=m).astype(float)


def state_cost(j, t, P, v):
    """Expected surviving value of target j when the weapons in t strike it."""
    return v[j] * math.prod(1 - P[i, j] for i in t)


def wta_value(assign, P, v):
    """Objective of an assignment given as assign[i] = target of weapon i."""
    return sum(state_cost(j, [i for i, a in enumerate(assign) if a == j], P, v) for j in range(len(v)))


def initial_columns(P, v, k_best=3):
    """Greedy plan (each weapon goes where it removes the most surviving value), its one-weapon
    moves (weapon i leaves its target, joins one of its k_best most valuable other targets),
    and the empty state of every target.
    The moves give the first LPs real alternatives; the greedy plan alone is a degenerate vertex
    that column generation leaves very slowly (stand-in for the paper's convex initialization)."""
    n, m = P.shape
    surviving, states = v.copy(), [frozenset() for _ in range(m)]
    for i in range(n):
        j = int(np.argmax(surviving * P[i]))
        surviving[j] *= 1 - P[i, j]
        states[j] |= {i}
    best_targets = [[k for k in np.argsort(-surviving * P[i]) if i not in states[k]][:k_best] for i in range(n)]
    moves = [(k, states[k] | {i}) for i in range(n) for k in best_targets[i]]
    moves += [(j, t - {i}) for j, t in enumerate(states) for i in t]
    return list(enumerate(states)) + moves + [(j, frozenset()) for j in range(m)]


# ---------- master LP ----------

class Master:
    """Restricted master LP, kept alive inside HiGHS:
        min q.z   s.t.  each weapon in at most one column (<= 1: firing a weapon never hurts,
                        so the optimum is unchanged), each target in exactly one column, z >= 0.
    Columns are only appended, so every re-solve starts from the previous basis (warm start).
    self.pool = {(j, t): cost}, in the same order as the HiGHS columns."""

    def __init__(self, n, m):
        self.n, self.pool = n, {}
        self.lp = highspy.Highs()
        self.lp.setOptionValue("output_flag", False)
        for r in range(n + m):                       # rows 0..n-1: weapons, rows n..n+m-1: targets
            self.lp.addRow(-highspy.kHighsInf if r < n else 1.0, 1.0, 0, [], [])

    def add(self, j, t, cost):
        rows = np.array(sorted(t) + [self.n + j], dtype=np.int32)
        self.lp.addCol(cost, 0.0, highspy.kHighsInf, len(rows), rows, np.ones(len(rows)))
        self.pool[(j, t)] = cost

    def forbid(self, forbidden):
        """Bound to 0 every column using a forbidden (weapon, target) pair, free the others.
        The empty state of each target is never forbidden, so the LP stays feasible."""
        ub = [0.0 if any((i, j) in forbidden for i in t) else highspy.kHighsInf for j, t in self.pool]
        K = len(ub)
        self.lp.changeColsBounds(K, np.arange(K, dtype=np.int32), np.zeros(K), np.array(ub))

    def solve(self):
        """Returns LP value, z, weapon duals lam (<= 0), target duals mu."""
        self.lp.run()
        sol = self.lp.getSolution()
        y = np.array(sol.row_dual)
        return self.lp.getObjectiveValue(), np.array(sol.col_value), y[:self.n], y[self.n:]


# ---------- pricing ----------

def best_subset(A, a, w):
    """Exact  min_{x in {0,1}^k}  A exp(-a.x) + w.x   (a, w > 0)  by depth-first branch and bound.
    Item i saves at most A (1 - exp(-a_i)) <= A a_i, so items with A a_i <= w_i never pay: dropped.
    Bound = continuous relaxation, solved greedily: take items by increasing w_i / a_i while the
    marginal gain A exp(-S) a_i beats the cost w_i; the last item taken may be fractional."""
    order = sorted((i for i in range(len(a)) if A * a[i] > w[i]), key=lambda i: w[i] / a[i])
    A, a, w = float(A), [float(a[i]) for i in order], [float(w[i]) for i in order]  # plain floats: fast
    best = [A, []]                                    # value, chosen positions (empty set first)

    def relaxation(k, S, W):
        for i in range(k, len(a)):
            gain = A * math.exp(-S) * a[i]
            if gain <= w[i]:
                break
            frac = min(1.0, math.log(gain / w[i]) / a[i])
            S, W = S + frac * a[i], W + frac * w[i]
        return A * math.exp(-S) + W

    def dfs(k, S, W, chosen):
        if A * math.exp(-S) + W < best[0]:
            best[:] = [A * math.exp(-S) + W, chosen]
        if k < len(a) and relaxation(k, S, W) < best[0] - 1e-12:
            dfs(k + 1, S + a[k], W + w[k], chosen + [k])   # take item k
            dfs(k + 1, S, W, chosen)                       # skip item k

    dfs(0, 0.0, 0.0, [])
    return [order[k] for k in best[1]], best[0]


def price(j, lam, mu_j, P, v, allowed):
    """Column of target j with the most negative reduced cost
        v_j exp(sum_{i in t} s_ij) - sum_{i in t} lam_i - mu_j,   s_ij = ln(1 - P_ij) < 0.
    A weapon with lam_i >= 0 lowers both terms, so it is always taken.
    The others (lam_i < 0) trade a survival reduction against a cost -lam_i: solved by best_subset."""
    s = np.log(1 - P[:, j])
    take = [i for i in allowed if lam[i] >= 0]
    rest = [i for i in allowed if lam[i] < 0]
    extra, value = best_subset(v[j] * np.exp(s[take].sum()), -s[rest], -lam[rest])
    t = frozenset(take + [rest[k] for k in extra])
    return t, value - lam[take].sum() - mu_j


def column_generation(master, P, v, forbidden):
    """Alternate master LP and pricing until no column has negative reduced cost."""
    n, m = P.shape
    master.forbid(forbidden)
    while True:
        lp, z, lam, mu = master.solve()
        new = []
        for j in range(m):
            allowed = [i for i in range(n) if (i, j) not in forbidden]
            t, rc = price(j, lam, mu[j], P, v, allowed)
            if rc < -EPS and (j, t) not in master.pool:
                new.append((j, t))
        if not new:
            return lp, z
        for j, t in new:
            master.add(j, t, state_cost(j, t, P, v))


# ---------- primal heuristic ----------

def solve_residual(candidates, pool, weapons, targets):
    """Pick candidate columns of minimum cost: each free target exactly once, each free weapon
    at most once. Small exact IP (MILP here) -> the piece to swap for a QUBO / quantum solver."""
    if not targets:
        return []
    A = [[i in t for j, t in candidates] for i in weapons] + [[j == g for j, t in candidates] for g in targets]
    lower = [0] * len(weapons) + [1] * len(targets)
    res = milp([pool[col] for col in candidates], constraints=LinearConstraint(np.array(A, float), lower, 1),
               integrality=np.ones(len(candidates)), bounds=Bounds(0, 1))
    return [candidates[k] for k in np.flatnonzero(res.x > 0.5)] if res.status == 0 else None


def primal_heuristic(pool, z, n, m):
    """Freeze the columns at 1 in the LP solution, solve the leftover problem over the pool
    (paper section 2.5). Returns a full plan (list of columns) or None."""
    cols = list(pool)
    frozen = [cols[k] for k in range(len(cols)) if z[k] > 1 - EPS]
    used_w = set().union(*(t for j, t in frozen))
    used_t = {j for j, t in frozen}
    candidates = [(j, t) for j, t in cols if j not in used_t and not (t & used_w)]
    rest = solve_residual(candidates, pool, [i for i in range(n) if i not in used_w],
                          [j for j in range(m) if j not in used_t])
    return None if rest is None else frozen + rest


# ---------- branching and tree search ----------

def branch_pair(pool, z, n, m):
    """Most fractional x_ij = sum of z over the columns of target j containing weapon i.
    None if x is integral (then z is integral too)."""
    x = np.zeros((n, m))
    for zk, (j, t) in zip(z, pool):
        x[list(t), j] += zk
    i, j = np.unravel_index(np.argmin(np.abs(x - 0.5)), x.shape)
    return (int(i), int(j)) if EPS < x[i, j] < 1 - EPS else None


def solve_wta(P, v):
    """Branch-and-price. A node is the set of forbidden (weapon, target) pairs:
    branch x_ij = 0 forbids (i, j); branch x_ij = 1 forbids (i, k) for every other k.
    Returns optimal value, assignment (weapon -> target), number of nodes."""
    n, m = P.shape
    master = Master(n, m)
    for j, t in dict.fromkeys(initial_columns(P, v)):            # dict.fromkeys drops duplicates
        master.add(j, t, state_cost(j, t, P, v))
    pool = master.pool
    best_value, best_plan, nodes, stack = np.inf, None, 0, [frozenset()]
    while stack:
        forbidden = stack.pop()
        nodes += 1
        lp, z = column_generation(master, P, v, forbidden)
        if lp >= best_value - EPS:                       # bound cannot improve: prune
            continue
        plan = primal_heuristic(pool, z, n, m)
        if plan is not None and sum(pool[col] for col in plan) < best_value:
            best_value, best_plan = sum(pool[col] for col in plan), plan
        pair = branch_pair(pool, z, n, m)
        if pair is not None:
            i, j = pair
            stack += [forbidden | {(i, k) for k in range(m) if k != j}, forbidden | {(i, j)}]
    # a weapon the plan leaves unused (allowed by the <= rows) is fired anyway: it can only help
    assign = [next((j for j, t in best_plan if i in t), int(np.argmax(v * P[i]))) for i in range(n)]
    return wta_value(assign, P, v), assign, nodes


if __name__ == "__main__":
    P, v = random_instance(20, 20, seed=1)
    value, assign, nodes = solve_wta(P, v)
    print(f"optimal value {value:.4f}  nodes {nodes}  assignment {assign}")
