"""Tests of the instance generator."""
import numpy as np

from wta_bpc import bpc
from wta_bpc.instances import fractional_instances, root_is_fractional, sparse_instance


def test_sparse_instance_shape_and_density():
    P, v = sparse_instance(20, 30, density=0.3, rng=np.random.default_rng(0))
    assert P.shape == (20, 30) and v.shape == (30,)
    assert 0.2 < np.mean(P > 0) < 0.4                       # about 30% of the pairs are effective
    assert P.max() < 0.9 and v.min() >= 1 and v.max() <= 10


def test_known_instances():
    """The 4x4 example has a fractional root LP, the 3x2 mini-WTA an integral one."""
    P_sparse = np.array([[0.0, 0.0, 0.0, 0.9],
                         [0.0, 0.6, 0.7, 0.6],
                         [0.0, 0.6, 0.9, 0.0],
                         [0.9, 0.9, 0.0, 0.0]])
    assert root_is_fractional(P_sparse, np.array([4.0, 9.0, 6.0, 10.0]))
    P_mini = np.array([[0.5, 0.2], [0.4, 0.6], [0.3, 0.5]])
    assert not root_is_fractional(P_mini, np.array([10.0, 6.0]))


def test_fractional_instances_need_branching():
    """Every generated instance has a fractional root, so the tree has more than one node."""
    instances, tries = fractional_instances(10, 10, density=0.5, count=3, seed=0)
    assert len(instances) == 3 and tries >= 3
    for P, v in instances:
        assert root_is_fractional(P, v)
        _, _, nodes = bpc.solve_wta(P, v)
        assert nodes > 1
