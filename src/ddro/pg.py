from __future__ import annotations

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
    """

    def __init__(
        self,
        nf: int,
        nc: int,
        sd: np.ndarray,
        oc: np.ndarray,
        tc: np.ndarray,
        cf: np.ndarray,
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
        """
        self.nf = nf
        self.nc = nc
        self.sd = sd
        self.oc = oc
        self.tc = tc
        self.cf = cf

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
