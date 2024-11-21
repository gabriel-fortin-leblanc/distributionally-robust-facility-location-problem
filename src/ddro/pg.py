from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class FLP:
    """
    A class representing the Facility Location Problem.

    Attributes
    ----------
    nf : int
        Number of possible locations for building facilities
        (Number-Facilities)
    nc : int
        Number of customer sites (Number-Customers)
    sd : array-like
        The finite support values the demand takes (Support-Demand)
    oc : array-like[nf]
        The costs for opening sites. Must be of size `nf`. (Opening-Cost)
    tc : array-like[nf, nc]
        The costs for transporting one item from facility `i` to customer site
        `j`. Must be a matrix of size `nf`x`nc`. (Transportation-Cost)
    cf : array-like[nf]
        The capacity of the facilities. Must be of size `nf`.
        (Capacity-Facility)
    rc : array-like[nc]
        The revenue for selling one item in costumer sites. (Revenue-Costumer)
    pc : array-like[nc]
        The penalty for not responding to the demands of costumer sites.
        (Penalty-Costumer)
    """

    def __init__(
        self,
        nf: int,
        nc: int,
        sd: np.ndarray,
        oc: np.ndarray,
        tc: np.ndarray,
        cf: np.ndarray,
        rc: np.ndarray,
        pc: np.ndarray,
    ):
        """
        Build the object.

        Parameters
        ----------
        nf : int
            Number of possible locations for building facilities
            (Number-Facilities)
        nc : int
            Number of customer sites (Number-Customers)
        sd : array-like
            The finite support values the demand takes (Support-Demand)
        oc : array-like[nf]
            The costs for opening sites. Must be of size `nf`. (Opening-Cost)
        tc : array-like[nf, nc]
            The costs for transporting one item from facility `i` to customer
            sites `j`. Must be a matrix of size `nf`x`nc`.
            (Transportation-Cost)
        cf : array-like[nf]
            The capacity of the facilities. Must be of size `nf`.
            (Capacity-Facility)
        rc : array-like[nc]
            The revenue for selling one item in costumer sites.
            (Revenue-Costumer)
        pc : array-like[nc]
            The penalty for not responding to the demands of costumer sites.
            (Penalty-Costumer)
        """
        self.nf = nf
        self.nc = nc
        self.sd = sd
        self.oc = oc
        self.tc = tc
        self.cf = cf
        self.rc = rc
        self.pc = pc

    @property
    def nf(self):
        return self._nf

    @nf.setter
    def nf(self, value):
        if type(value) is not int:
            raise TypeError("'nf' must be a integer")
        if value <= 0:
            raise ValueError("'nf' must be strictly positive")
        self._nf = value

    @property
    def nc(self):
        return self._nc

    @nc.setter
    def nc(self, value):
        if type(value) is not int:
            raise TypeError("'nc' must be a integer")
        if value <= 0:
            raise ValueError("'nc' must be strictly positive")
        self._nc = value

    @property
    def sd(self):
        return self._sd

    @sd.setter
    def sd(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'sd' must be a Numby Array")
        if value.ndim != 1 or np.squeeze(value).ndim != 1:
            raise ValueError("'sd' must have only one dimension")
        self._sd = value

    @property
    def oc(self):
        return self._oc

    @oc.setter
    def oc(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'oc' must be a Numby Array")
        if value.ndim != 1 or np.squeeze(value).ndim != 1:
            raise ValueError("'oc' must have only one dimension")
        if value.shape[0] != self.nf:
            raise ValueError("'oc' must be of size 'nf'")
        if (value <= 0).any():
            raise ValueError("'oc' must contain strictly positive values")
        self._oc = value

    @property
    def tc(self):
        return self._tc

    @tc.setter
    def tc(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'tc' must be a Numby Array")
        if value.ndim != 2 or np.squeeze(value).ndim != 2:
            raise ValueError("'tc' must have only two dimensions")
        if value.shape != (self.nf, self.nc):
            raise ValueError("'tc' must be of size ['nf', 'nc']")
        if (value <= 0).any():
            raise ValueError("'tc' must contain strictly positive values")
        self._tc = value

    @property
    def cf(self):
        return self._cf

    @cf.setter
    def cf(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'cf' must be a Numby Array")
        if value.ndim != 1 or np.squeeze(value).ndim != 1:
            raise ValueError("'cf' must have only one dimension")
        if value.shape[0] != self.nf:
            raise ValueError("'cf' must be of size 'nf'")
        if (value <= 0).any():
            raise ValueError("'cf' must contain strictly positive values")
        self._cf = value

    @property
    def rc(self):
        return self._rc

    @rc.setter
    def rc(self, value):
        self._rc = value

    @property
    def pc(self):
        return self._pc

    @pc.setter
    def pc(self, value):
        self._pc = value


def flp_generator(
    nf: int = 10,
    nc: int = 20,
    sd: np.ndarray = np.arange(1, 101),
    oc: Optional[np.ndarray] = None,
    fp: Optional[np.ndarray] = None,
    csp: Optional[np.ndarray] = None,
    cf: Optional[np.ndarray] = None,
    rc: Optional[np.ndarray] = None,
    pc: Optional[np.ndarray] = None,
    tcf: Optional[float] = None,
    dist: Callable[[np.ndarray, np.ndarray], float] = (
        lambda x, y: np.linalg.norm(x - y)
    ),
):
    """
    Generate a FLP problem (See FLP).

    Parameters
    ----------
    nf : int
        Number of possible locations for building facilities. Default: 10
        (Number-Facilities)
    nc : int
        Number of customer sites. Default: 20 (Number-Customers)
    sd : array-like
        The finite support of values the demand takes (Support-Demand).
        Default: [1, ..., 100]
    oc : array-like[nf] or a generator of positive values
        The costs for opening sites. Must be of size `nf`. Default: Uniform
        on [5000, 10000). (Opening-Cost)
    fp : array-like[nf, 2] or a generator
        The position of the possible facilities. Default:
        Uniform on [-10, 10). (Facility-Position)
    csp : array-like[nc, 2] or a generator
        The position of the customer sites. Default: Uniform on [-10, 10).
        (Customer-Site-Position)
    cf : array-like[nf] or a generator of positive values
        The capacity of the facilities. Must be of size `nf`. Default: Uniform
        on [10, 20). (Capacity-Facility)
    rc : array-like[nc]
        The revenue for selling one item in a specific costumer site. Must be
        of size `nc`. Default: 150 * `nc`.
    pc : array-like[nc]
        The penalty for not reponding to the demands. Must be of size `nc`.
        Default: 225 * `nc`.
    tcf : float or a generator of positive values
        The factor for transportation costs. It is proportional to the
        distance between a facility and a customer site. Default: Uniform
        on [1, 3). (Transportation-Cost-Factor)
    dist : distance function
        The distance used between facilities and customer sites. Default:
        euclidian distance.

    Returns
    -------
    FLP
        A facility location problem.
    """
    if nf <= 0:
        raise ValueError("nf must be strictly postive")
    if nc <= 0:
        raise ValueError("nc must be strictly postive")

    # Compute the transportation costs
    _fp = (
        fp
        if fp is not None
        else np.random.uniform(low=-10, high=10, size=(nf, 2))
    )
    _csp = (
        csp
        if csp is not None
        else np.random.uniform(low=-10, high=10, size=(nc, 2))
    )
    _tcf = tcf if tcf is not None else np.random.uniform(low=1, high=3)
    tc = np.empty((nf, nc))
    for i in range(nf):
        for j in range(nc):
            tc[i, j] = _tcf * dist(_fp[i], _csp[j])

    return FLP(
        nf=nf,
        nc=nc,
        sd=sd,
        oc=(
            oc
            if oc is not None
            else np.random.uniform(low=5000, high=10000, size=nf)
        ),
        tc=tc,
        cf=(
            cf
            if cf is not None
            else np.random.uniform(low=10, high=20, size=nf)
        ),
        pc=(pc if pc is not None else np.repeat(225, nc)),
        rc=(rc if rc is not None else np.repeat(150, nc)),
    )
