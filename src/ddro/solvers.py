from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .pg import FLP


class Solver(ABC):
    """
    Abstract class for solver. A solver has a attribute `y` that contains the
    decision variables and a function `solve` that solve the problem.
    """

    @property
    def y(self):
        if not hasattr(self, "_y"):
            raise AttributeError("y does not exist. Must solve problem first.")
        return self._y

    @y.setter
    def y(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("y must be a Numpy array.")
        self._y = value

    @abstractmethod
    def solve(flp: FLP) -> bool:
        """
        Solve the problem `flp` and return a boolean depending on the success.
        """
        raise NotImplementedError("solve is not implemented")


class BASSolver(Solver):
    """
    Implementation of the Basciftci, Ahmed and Shen solver from
    `Distributionally robust facility location problem under decision-dependent
    stochastic demand`.

    Attributes
    ----------
    TODO
    """

    def __init__(
        self,
        mu_bar: Optional[np.ndarray] = None,
        eps_mu: Optional[np.ndarray] = None,
        lbd_mu: Optional[np.ndarray] = None,
        sig_bar: Optional[np.ndarray] = None,
        eps_under_sig: Optional[np.ndarray] = None,
        eps_upper_sig: Optional[np.ndarray] = None,
        lbs_sig: Optional[np.ndarray] = None,
        nf: Optional[int] = None,
        nc: Optional[int] = None,
    ):
        """
        Initiate a BASSolver object with respect to hyperparameters.

        Parameters
        ----------
        mu_bar : array-like[nc], optional
            Factor from decision variable `y` for computing the 'mean' of
            expected values of the uncertainty set. Default: Uniform over
            [20, 40).
        eps_mu : array-like[nc], optional
            Upper bound on the difference between expected values and the
            'mean' of expected values on the uncertainty set. Default: 0.
        lbd_mu : array-like[nc, nf], optional
            Impact from opened facilities on 'mean' of expected values of the
            uncertainty set.
        sig_bar : array-like[nc], optional
            Factor from decision variable `y` for computing the 'mean' of
            variances of the uncertainty set. Default: Equals to `mu_bar`.
        eps_under_sig : array-like[nc], optional
            Lower bound on the second moments of the uncertainty set.
            Default: 1.
        eps_upper_sig : array-like[nc], optional
            Upper bound on the second moments of the uncertainty set.
            Default: 1.
        lbd_sig : array-like[nc, nf], optional
            Impact from opened facilities on 'mean' of variances of the
            uncertainty set.
        nf : int, optional
            Number of possible locations for building facilities. Default: 10.
        nc : int, optional
            Number of costumer sites. Default: 20.
        """
        pass

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
    def mu_bar(self):
        return self._mu_bar

    @mu_bar.setter
    def mu_bar(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'mu_bar' must be a Numpy array")
        self._mu_bar = value

    @property
    def eps_mu(self):
        return self._eps_mu

    @eps_mu.setter
    def mu_eps(self, value):
        self._eps_mu = value

    @property
    def lbd_mu(self):
        return self._lbd_mu

    @lbd_mu.setter
    def lbd_mu(self, value):
        self._lbd_mu = value

    @property
    def sig_bar(self):
        return self._sig_bar

    @sig_bar.setter
    def sig_bar(self, value):
        self._sig_bar = value

    @property
    def eps_under_sig(self):
        return self._eps_under_sig

    @eps_under_sig.setter
    def eps_under_sig(self, value):
        self._eps_under_sig = value

    @property
    def eps_upper_sig(self):
        return self._eps_upper_sig

    @eps_upper_sig.setter
    def eps_upper_sig(self, value):
        self._eps_upper_sig = value

    @property
    def lbd_sig(self):
        return self._lbd_sig

    @lbd_sig.setter
    def lbd_sig(self, value):
        self._lbd_sig = value

    def solve(flp: FLP) -> bool:
        # TODO: To implement
        pass

    def _cvn(self, flp: FLP):
        """
        Convert notations.
        """
        J = flp.nf
        I = flp.nc
        f = flp.oc
        c = flp.tc
        C = flp.cf
        xi = flp.sd
        return (J, I, f, c, C, xi)