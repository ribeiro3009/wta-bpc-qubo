"""Tests of the benchmark save / load of step-2.5 problems."""
from pathlib import Path

import pytest

from wta_bpc import benchmark, bpc

BENCHMARK = Path(__file__).parent.parent / "experiments" / "benchmark"


def test_round_trip(sparse_residual, tmp_path):
    candidates, pool, weapons, targets = sparse_residual
    optimum = sum(pool[c] for c in bpc.solve_residual(candidates, pool, weapons, targets))
    d = benchmark.problem_to_dict("sparse4x4", candidates, pool, weapons, targets, optimum)
    benchmark.save([d], tmp_path / "b.json")
    [loaded] = benchmark.load(tmp_path / "b.json")
    assert loaded["variables"] == 14 and loaded["onehot_states"] == 100
    c2, pool2, w2, t2 = benchmark.problem_from_dict(loaded)
    assert c2 == list(candidates)
    assert pool2 == {c: pool[c] for c in candidates}         # only the candidate columns are saved
    assert sum(pool2[c] for c in bpc.solve_residual(c2, pool2, w2, t2)) == pytest.approx(optimum)


@pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmark not exported")
def test_exported_optima_are_correct():
    """For every exported problem, the MILP on the loaded data gives the stored optimum."""
    for path in sorted(BENCHMARK.glob("*.json")):
        for d in benchmark.load(path):
            candidates, pool, weapons, targets = benchmark.problem_from_dict(d)
            cols = bpc.solve_residual(candidates, pool, weapons, targets)
            assert sum(pool[c] for c in cols) == pytest.approx(d["optimum"]), d["name"]
