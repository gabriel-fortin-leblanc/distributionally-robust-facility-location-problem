from typing import Union

import gurobipy as gp
import numpy as np

from .pg import FLP


def benchmark(
    ys: Union[list[np.ndarray], np.ndarray],
    flp: FLP,
    prob: np.ndarray = None,
    N: int = 250,
):
    """
    Stochasticly test 'n' location decision with respect to a FLP.

    Parameters
    ----------
    ys : array-like[n, flp.nf] or [array-like[flp.nf]]
        'n' different location decisions.
    flp : FLP
        The facility location problem related to the decisions.
    prob : array-like[flp.nc, len(flp.sd)]
        The probabilities for each outcome of the support. Default: uniform
        probability is taken.
    N : int
        The number of tests for each decision. Default: 250.
    """
    if type(ys) is list:
        if len(ys) == 0:
            raise ValueError("'ys' must be an non-empty list")
        for y in ys:
            if type(y) is not np.ndarray:
                raise TypeError("'ys' must be a list of Numpy array")
        ys = np.array(ys)
    elif type(ys) is np.ndarray:
        if ys.ndim == 2 and ys.shape[1] != flp.nf:
            raise ValueError(
                "The decisions 'ys' must be compatible with the 'flp'"
            )
        elif ys.ndim == 1 and ys.shape[0] != flp.nf:
            raise ValueError(
                "The decisions 'ys' must be compatible with the 'flp'"
            )
        elif ys.ndim > 2:
            raise ValueError("'ys' must be of at most two dimensions")
    else:
        raise TypeError("'ys' must be a Numpy array or a list of Numpy array")
    if ys.ndim == 1:
        ys = ys.reshape((1, -1))
        ny = 1
    else:
        ny = ys.shape[0]

    if N <= 0:
        raise ValueError("'N' must be strictly positive")

    if prob is None:
        prob = np.full((flp.nc, flp.sd.shape[0]), 1 / flp.sd.shape[0])
    elif (prob < 0).any() or not np.isclose(prob.sum(1), 1).all():
        raise ValueError("'prob' must represent probabilities")
    cdf = np.cumsum(prob, axis=1)

    objs = np.empty((N, ny))
    ss = np.empty((N, ny))
    us = np.random.uniform(size=(N, flp.nc))
    ds = np.empty((N, flp.nc))
    for k in range(N):
        d_idx = np.argmax(us[k][:, np.newaxis] < cdf, axis=1)
        ds[k] = flp.sd[d_idx]

    for n in range(ny):
        y = ys[n]
        model = gp.Model()
        x = model.addMVar((N, flp.nf, flp.nc))
        s = model.addMVar((N, flp.nc))
        obj = N * flp.oc @ y
        obj += (flp.tc[np.newaxis, :, :] * x).sum()
        obj += (flp.pc[np.newaxis, :] * s - flp.rc[np.newaxis, :] * ds).sum()
        obj /= N
        model.setObjective(obj)
        model.addConstr(x.sum(1) + s == ds)
        model.addConstr(
            x
            <= flp.cf[np.newaxis, :, np.newaxis] * y[np.newaxis, :, np.newaxis]
        )
        model.addConstr(s >= 0)
        model.addConstr(x >= 0)
        model.optimize()
        _x = x.x
        _s = s.x
        objs[:, n] = np.full((N,), flp.oc @ y)
        objs[:, n] += (_x * flp.tc[np.newaxis, :, :]).sum(2).sum(1)
        objs[:, n] += (
            flp.pc[np.newaxis, :] * _s - flp.rc[np.newaxis, :] * ds
        ).sum(1)
        ss[:, n] = _s.sum(1)

    return objs, ss
