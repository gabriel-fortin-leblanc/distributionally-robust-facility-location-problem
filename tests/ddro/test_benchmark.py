import numpy as np

from ddro.benchmark import benchmark
from ddro.pg import FLP


def test_benchmark():
    flp = FLP(
        2,
        3,
        np.array([1, 2, 3]),
        np.array([1, 2]),
        np.array([[1, 2, 3], [3, 4, 5]]),
        np.array([1, 2]),
        np.array([1, 2, 3]),
        np.array([2, 2, 2]),
    )
    y = np.array([0, 1])
    objs, ss = benchmark(y, flp)
    assert objs.shape[0] == 250
    assert ss.shape[0] == 250

    y = [np.array([0, 1]), np.array([1, 0])]
    objs, ss = benchmark(y, flp)
    assert objs.shape == (250, 2)
    assert ss.shape == (250, 2)
