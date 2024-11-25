from abc import ABC, abstractmethod
from warnings import deprecated
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
            raise AttributeError(
                "'y' does not exist. Must solve problem first."
            )
        return self._y

    @y.setter
    def y(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'y' must be a Numpy array.")
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
    stochastic demand`. It extends `Solver`.

    Attributes
    ----------
    mu_bar : array-like[nc]
        Factor from decision variable `y` for computing the 'mean' of
        expected values of the uncertainty set.
    eps_mu : array-like[nc]
        Upper bound on the difference between expected values and the
        'mean' of expected values on the uncertainty set.
    lbd_mu : array-like[nc, nf]
        Impact from opened facilities on 'mean' of expected values of the
        uncertainty set.
    sig_bar : array-like[nc]
        Factor from decision variable `y` for computing the 'mean' of
        variances of the uncertainty set.
    eps_lower_sig : array-like[nc]
        Lower bound on the second moments of the uncertainty set.
    eps_upper_sig : array-like[nc]
        Upper bound on the second moments of the uncertainty set.
    lbd_sig : array-like[nc, nf]
        Impact from opened facilities on 'mean' of variances of the
        uncertainty set.
    nf : int
        Number of possible locations for building facilities.
    nc : int
        Number of costumer sites.

    References
    ----------
    [1] Basciftci, B., Ahmed, S., & Shen, S. (2021). Distributionally robust
    facility location problem under decision-dependent stochastic demand.
    European Journal of Operational Research, 292(2), 548–561.
    https://doi.org/10.1016/j.ejor.2020.11.002

    """

    def __init__(
        self,
        mu_bar: Optional[np.ndarray] = None,
        eps_mu: Optional[np.ndarray] = None,
        lbd_mu: Optional[np.ndarray] = None,
        sig_bar: Optional[np.ndarray] = None,
        eps_lower_sig: Optional[np.ndarray] = None,
        eps_upper_sig: Optional[np.ndarray] = None,
        lbd_sig: Optional[np.ndarray] = None,
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
            uncertainty set. Default: will be computed during the solve
            phase. transpose(exp(-transportation costs/25))
        sig_bar : array-like[nc], optional
            Factor from decision variable `y` for computing the 'mean' of
            variances of the uncertainty set. Default: Equals to `mu_bar`.
        eps_lower_sig : array-like[nc], optional
            Lower bound on the second moments of the uncertainty set.
            Default: 1.
        eps_upper_sig : array-like[nc], optional
            Upper bound on the second moments of the uncertainty set.
            Default: 1.
        lbd_sig : array-like[nc, nf], optional
            Impact from opened facilities on 'mean' of variances of the
            uncertainty set. Default: will be computed during the solve phase.
            transpose(exp(-transportation costs/25))
        nf : int, optional
            Number of possible locations for building facilities. Default: 10.
        nc : int, optional
            Number of costumer sites. Default: 20.
        """
        self.nf = nf if nf is not None else 10
        self.nc = nc if nc is not None else 20
        self.mu_bar = (
            mu_bar if mu_bar is not None
            else np.random.uniform(low=20, high-40, size=self.nc)
        )
        self.eps_mu = eps_mu if eps_mu is not None else 0
        self.lbd_mu = lbd_mu
        self.sig_bar = (
            sig_bar if sig_bar is not None
            else np.array(self.mu_bar, copy=True)
        )
        self.eps_lower_sig = eps_lower_sig if eps_lower_sig is not None else 1
        self.eps_upper_sig = eps_upper_sig if eps_upper_sig is not None else 1
        self.lbd_sig = lbd_sig


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
        if value.ndim != 1 or np.squeeze(value) != 1:
            raise ValueError("'mu_bar' must have only one dimension")
        if value.shape[0] != self.nc:
            raise ValueError("'mu_bar' must be of size 'nc'")
        self._mu_bar = value

    @property
    def eps_mu(self):
        return self._eps_mu

    @eps_mu.setter
    def mu_eps(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'mu_eps' must be a Numpy array")
        if value.ndim != 1 or np.squeeze(value) != 1:
            raise ValueError("'mu_eps' must have only one dimension")
        if value.shape[0] != self.nc:
            raise ValueError("'mu_eps' must be of size 'nc'")
        self._eps_mu = value

    @property
    def lbd_mu(self):
        return self._lbd_mu

    @lbd_mu.setter
    def lbd_mu(self, value):
        if value is None:
            self._lbd_mu
        else:
            if type(value) is not np.ndarray:
                raise TypeError("'lbd_mu' must be a Numpy array")
            if value.ndim != 2 or np.squeeze(value) != 2:
                raise ValueError("'lbd_mu' must have two dimensions")
            if value.shape[0] != self.nc or value.shape[1] != self.nf:
                raise ValueError("'lbd_mu' must be of size ['nc', 'nf']")
            self._lbd_mu = value

    @property
    def sig_bar(self):
        return self._sig_bar

    @sig_bar.setter
    def sig_bar(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'sig_bar' must be a Numpy array")
        if value.ndim != 1 or np.squeeze(value) != 1:
            raise ValueError("'sig_bar' must have only one dimension")
        if value.shape[0] != self.nc:
            raise ValueError("'sig_bar' must be of size 'nc'")
        self._sig_bar = value

    @property
    def eps_lower_sig(self):
        return self._eps_lower_sig

    @eps_lower_sig.setter
    def eps_lower_sig(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'eps_lower_sig' must be a Numpy array")
        if value.ndim != 1 or np.squeeze(value) != 1:
            raise ValueError("'eps_lower_sig' must have only one dimension")
        if value.shape[0] != self.nc:
            raise ValueError("'eps_lower_sig' must be of size 'nc'")
        self._eps_lower_sig = value

    @property
    def eps_upper_sig(self):
        return self._eps_upper_sig

    @eps_upper_sig.setter
    def eps_upper_sig(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'eps_upper_sig' must be a Numpy array")
        if value.ndim != 1 or np.squeeze(value) != 1:
            raise ValueError("'eps_upper_sig' must have only one dimension")
        if value.shape[0] != self.nc:
            raise ValueError("'eps_upper_sig' must be of size 'nc'")
        self._eps_upper_sig = value

    @property
    def lbd_sig(self):
        return self._lbd_sig

    @lbd_sig.setter
    def lbd_sig(self, value):
        if value is None:
            self.lbd_mu = value
        else:
            if type(value) is not np.ndarray:
                raise TypeError("'lbd_sig' must be a Numpy array")
            if value.ndim != 2 or np.squeeze(value) != 2:
                raise ValueError("'lbd_sig' must have two dimensions")
            if value.shape[0] != self.nc or value.shape[1] != self.nf:
                raise ValueError("'lbd_sig' must be of size ['nc', 'nf']")
            self._lbd_sig = value

    # The following comment is for using confusing variable names like J, I,
    # etc. without being harassed by flake8.
    # flake8: noqa: E741
    def solve(self, flp: FLP) -> bool:
        J, I, f, c, C, xi, r, p = self._cvn(flp)
        # TODO: To implement
        pass

    @deprecated("Deprecated inner function: don't need to convert variables")
    def _cvn(self, flp: FLP):
        """
        Convert notations to the ones used in the referenced article.
        """
        J = flp.nf
        I = flp.nc
        f = flp.oc
        c = flp.tc
        C = flp.cf
        xi = flp.sd
        r = flp.rc
        p = flp.pc
        return (J, I, f, c, C, xi, r, p)
