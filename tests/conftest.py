"""Shared test data (pytest fixtures): the hand-computed QUBO case and a real step-2.5 problem."""
import numpy as np
import pytest

from wta_bpc import bpc

F = frozenset


@pytest.fixture
def hand_case():
    """Case 3: free targets 1 (v = 9) and 2 (v = 6), free weapons 1 and 2, 6 candidate columns.
    With A = B = 10 the QUBO minimum is E = 4.2 at (1,{1}) + (2,{2})."""
    pool = {
        (1, F({1})): 3.6,
        (1, F({1, 2})): 1.44,
        (1, F()): 9.0,
        (2, F({2})): 0.6,
        (2, F({1})): 1.8,
        (2, F()): 6.0,
    }
    return list(pool), pool, [1, 2], [1, 2]


@pytest.fixture
def sparse_residual(monkeypatch):
    """The residual problem that step 2.5 receives at the root of the sparse 4x4 instance
    (14 candidate columns), as (candidates, pool, weapons, targets)."""
    P = np.array([[0.0, 0.0, 0.0, 0.9],
                  [0.0, 0.6, 0.7, 0.6],
                  [0.0, 0.6, 0.9, 0.0],
                  [0.9, 0.9, 0.0, 0.0]])
    v = np.array([4.0, 9.0, 6.0, 10.0])

    # Run the branch-and-price, recording the arguments given to solve_residual.
    calls = []
    original = bpc.solve_residual

    def spy(candidates, pool, weapons, targets):
        calls.append((candidates, dict(pool), weapons, targets))
        return original(candidates, pool, weapons, targets)

    monkeypatch.setattr(bpc, "solve_residual", spy)
    bpc.solve_wta(P, v)
    return calls[0]                                    # the call made at the root node
