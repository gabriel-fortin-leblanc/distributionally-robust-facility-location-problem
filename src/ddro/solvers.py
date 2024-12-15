import warnings
from abc import ABC, abstractmethod
from typing import Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .pg import FLP


class Solver(ABC):
    """
    Abstract class for solver. A solver has a attribute `y` that contains the
    decision variables and a function `solve` that solve the problem.
    """

    @abstractmethod
    def solve(flp: FLP) -> bool:
        """
        Solve the problem `flp` and return a boolean depending on the success.
        """
        raise NotImplementedError("solve is not implemented")

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
        self._y[self._y < 0.3] = 0
        self._y[self._y > 0.7] = 1

    @property
    def obj(self):
        if not hasattr(self, "_obj"):
            raise AttributeError(
                "'obj' does not exist. Must solve problem first."
            )
        return self._obj

    @obj.setter
    def obj(self, value):
        if type(value) is not float:
            raise TypeError("'obj' must be a float")
        self._obj = value


class PSolver(Solver):
    """
    A plain (non-robust) solver for the FLP. It has no hyperparameters.
    """

    def solve(self, flp: FLP) -> bool:
        """
        Solve the problem `flp` and return a boolean depending on the success.
        """
        d = flp.sd.mean()
        model = gp.Model()
        y = model.addMVar(flp.nf, vtype=GRB.BINARY)
        x = model.addMVar((flp.nf, flp.nc))
        s = model.addMVar(flp.nc)
        model.setObjective(
            flp.oc @ y + (flp.tc * x).sum() + (flp.pc * s - flp.rc * d).sum()
        )
        model.addConstr(x.sum(0) + s == d)
        model.addConstr(x <= (flp.cf * y)[:, np.newaxis])
        model.addConstr(x >= 0)
        model.addConstr(s >= 0)
        model.optimize()

        self.y = y.x
        self.obj = model.objVal
        return True


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
    delt1_upper : float
        An upper bound on the dual variable delta_1. See [1].
    delt2_upper : float
        An upper bound on the dual variable delta_2. See [2].
    gam1_upper : float
        An upper bound on the dual variable gamma_1. See [1].
    gam2_upper : float
        An upper bound on the dual variable gamma_2. See [2].

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
        delt1_upper: Optional[float] = 100000,
        delt2_upper: Optional[float] = 100000,
        gam1_upper: Optional[float] = 100000,
        gam2_upper: Optional[float] = 100000,
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
            mu_bar
            if mu_bar is not None
            else np.random.uniform(low=20, high=40, size=self.nc)
        )
        self.eps_mu = eps_mu if eps_mu is not None else np.zeros(self.nc)
        self.lbd_mu = lbd_mu
        self.sig_bar = (
            sig_bar
            if sig_bar is not None
            else np.array(self.mu_bar, copy=True)
        )
        self.eps_lower_sig = (
            eps_lower_sig if eps_lower_sig is not None else np.ones(self.nc)
        )
        self.eps_upper_sig = (
            eps_upper_sig if eps_upper_sig is not None else np.ones(self.nc)
        )
        self.lbd_sig = lbd_sig
        self.delt1_upper = delt1_upper
        self.delt2_upper = delt2_upper
        self.gam1_upper = gam1_upper
        self.gam2_upper = gam2_upper

    def solve(self, flp: FLP) -> bool:
        """
        Solve the problem `flp` and return a boolean depending on the success.
        """
        if flp.nc != self.nc:
            raise ValueError(
                "'flp' not compatible. Must have the same"
                "'nc' value than the solver"
            )
        if flp.nf != self.nf:
            raise ValueError(
                "'flp' not compatible. Must have the same"
                "'nf' value than the solver"
            )

        __lbd_mu = (
            self.lbd_mu if self.lbd_mu is not None else np.exp(-flp.tc.T / 25)
        )
        tmp = __lbd_mu.sum(1)
        __lbd_mu[tmp > 0] /= tmp[tmp > 0, np.newaxis]
        __lbd_sig = (
            self.lbd_sig
            if self.lbd_sig is not None
            else np.array(__lbd_mu, copy=True)
        )
        tmp = __lbd_sig.sum(1)
        __lbd_sig[tmp > 0] /= __lbd_sig.sum(1)[tmp > 0, np.newaxis]

        model = gp.Model("Facility Location Problem")
        # Define decision variables
        y = model.addMVar(self.nf, vtype=GRB.BINARY, name="Opening variable")
        alph = model.addMVar(self.nc)
        delt1 = model.addMVar(self.nc)
        delt2 = model.addMVar(self.nc)
        gam1 = model.addMVar(self.nc)
        gam2 = model.addMVar(self.nc)
        Delt1 = model.addMVar((self.nc, self.nf))
        Delt2 = model.addMVar((self.nc, self.nf))
        Gam1 = model.addMVar((self.nc, self.nf))
        Gam2 = model.addMVar((self.nc, self.nf))
        Psi1 = model.addMVar((self.nc, self.nf, self.nf))
        Psi2 = model.addMVar((self.nc, self.nf, self.nf))
        Y = model.addMVar((self.nf, self.nf))
        Theta = model.addMVar(self.nc)

        # Useful constant
        Lbd = -self.sig_bar[:, np.newaxis] * __lbd_sig + (self.mu_bar**2)[
            :, np.newaxis
        ] * (2 * __lbd_mu + __lbd_mu * __lbd_mu)
        # Add objective
        obj = flp.oc @ y
        for j in range(self.nc):
            obj += (
                alph[j]
                + delt1[j] * (self.mu_bar[j] + self.eps_mu[j])
                - delt2[j] * (self.mu_bar[j] - self.eps_mu[j])
            )
            for i in range(self.nf):
                obj += self.mu_bar[j] * (
                    __lbd_mu[j, i] * (Delt1[j, i] - Delt2[j, i])
                )
            obj += (self.sig_bar[j] + self.mu_bar[j] * self.mu_bar[j]) * (
                self.eps_upper_sig[j] * gam1[j]
                - self.eps_lower_sig[j] * gam2[j]
            )
            for i in range(self.nf):
                obj += Lbd[j, i] * (
                    self.eps_upper_sig[j] * Gam1[j, i]
                    - self.eps_lower_sig[j] * Gam2[j, i]
                )
            for l in range(1, self.nf):
                for m in range(l):
                    obj += (
                        2
                        * self.mu_bar[j]
                        * self.mu_bar[j]
                        * __lbd_mu[j, l]
                        * __lbd_mu[j, m]
                        * (
                            self.eps_upper_sig[j] * Psi1[j, l, m]
                            - self.eps_lower_sig[j] * Psi2[j, l, m]
                        )
                    )
        model.setObjective(obj)

        # Add constraints (17b)
        for j in range(self.nc):
            for k in range(flp.sd.shape[0]):
                # i == 0 for the following line
                model.addConstr(
                    alph[j]
                    + (delt1[j] - delt2[j]) * flp.sd[k]
                    + (gam1[j] - gam2[j]) * flp.sd[k] ** 2
                    >= (flp.pc[j] - flp.rc[j]) * flp.sd[k]
                    + (flp.cf * y * (flp.tc[:, j] - flp.pc[j])).sum()
                )
                for i in range(self.nf):
                    mask = flp.tc[:, j] < flp.tc[i, j]
                    model.addConstr(
                        alph[j]
                        + (delt1[j] - delt2[j]) * flp.sd[k]
                        + (gam1[j] - gam2[j]) * flp.sd[k] ** 2
                        >= (flp.tc[i, j] - flp.rc[j]) * flp.sd[k]
                        + (
                            flp.cf[mask]
                            * y[mask]
                            * (flp.tc[mask, j] - flp.tc[i, j])
                        ).sum()
                    )
        # Add McCormick envelopes constaints
        for j in range(self.nc):
            for i in range(self.nf):
                model.addConstrs(
                    cstr
                    for cstr in self._m1_constraints(
                        Delt1[j, i], delt1[j], y[i], 0, self.delt1_upper
                    )
                )
                model.addConstrs(
                    cstr
                    for cstr in self._m1_constraints(
                        Delt2[j, i], delt2[j], y[i], 0, self.delt2_upper
                    )
                )
                model.addConstrs(
                    cstr
                    for cstr in self._m1_constraints(
                        Gam1[j, i], gam1[j], y[i], 0, self.gam1_upper
                    )
                )
                model.addConstrs(
                    cstr
                    for cstr in self._m1_constraints(
                        Gam2[j, i], gam2[j], y[i], 0, self.gam2_upper
                    )
                )
                for m in range(self.nf):
                    model.addConstrs(
                        cstr
                        for cstr in self._m2_constraints(
                            Psi1[j, i, m],
                            gam1[j],
                            y[i],
                            y[m],
                            0,
                            self.gam1_upper,
                        )
                    )
                    model.addConstrs(
                        cstr
                        for cstr in self._m2_constraints(
                            Psi2[j, i, m],
                            gam2[j],
                            y[i],
                            y[m],
                            0,
                            self.gam2_upper,
                        )
                    )
        # Add non-negative constaints
        model.addConstr(delt1 >= 0)
        model.addConstr(delt2 >= 0)
        model.addConstr(gam1 >= 0)
        model.addConstr(gam2 >= 0)
        # Add valid constaints
        tmp = 0
        for l in range(1, self.nf):
            for m in range(l):
                tmp += __lbd_mu[:, l] * __lbd_mu[l, m] * Y[l, m]
        for j in range(self.nc):
            model.addConstr(
                Theta[j]
                == self.sig_bar[j]
                + self.mu_bar[j] ** 2
                + (Lbd[j] * y).sum()
                + 2 * self.mu_bar[j] ** 2 * tmp[j]
            )
            model.addConstr(
                flp.sd[0] * flp.sd[1]
                - (flp.sd[0] + flp.sd[1])
                * (
                    self.mu_bar[j] * (1 + (__lbd_mu[j] * y).sum())
                    - self.eps_mu[j]
                )
                + Theta[j] * self.eps_upper_sig[j]
                >= 0
            )
            model.addConstr(
                flp.sd[-2] * flp.sd[-1]
                - (flp.sd[-2] + flp.sd[-1])
                * (
                    self.mu_bar[j] * (1 + (__lbd_mu[j] * y).sum())
                    - self.eps_mu[j]
                )
                + Theta[j] * self.eps_upper_sig[j]
                >= 0
            )
            model.addConstr(
                -flp.sd[0] * flp.sd[-1]
                + (flp.sd[0] + flp.sd[-1])
                * (
                    self.mu_bar[j] * (1 + (__lbd_mu[j] * y).sum())
                    + self.eps_mu[j]
                )
                - Theta[j] * self.eps_lower_sig[j]
                >= 0
            )
        for l in range(1, self.nf):
            for m in range(l):
                model.addConstrs(
                    cstr
                    for cstr in self._m1_constraints(Y[l, m], y[l], y[m], 0, 1)
                )

        model.optimize()
        self.y = y.x
        self.obj = model.objVal
        return True

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
    def delt1_upper(self):
        return self._delt1_upper

    @delt1_upper.setter
    def delt1_upper(self, value):
        if type(value) is not float and type(value) is not int:
            raise TypeError("'delt1_upper' must be a float or a int")
        if value <= 0:
            raise ValueError("'delt1_upper' mustbe strictly positive")
        self._delt1_upper = value

    @property
    def delt2_upper(self):
        return self._delt2_upper

    @delt2_upper.setter
    def delt2_upper(self, value):
        if type(value) is not float and type(value) is not int:
            raise TypeError("'delt2_upper' must be a float or a int")
        if value <= 0:
            raise ValueError("'delt2_upper' mustbe strictly positive")
        self._delt2_upper = value

    @property
    def gam1_upper(self):
        return self._gam1_upper

    @gam1_upper.setter
    def gam1_upper(self, value):
        if type(value) is not float and type(value) is not int:
            raise TypeError("'gam1_upper' must be a float or a int")
        if value <= 0:
            raise ValueError("'gam1_upper' mustbe strictly positive")
        self._gam1_upper = value

    @property
    def gam2_upper(self):
        return self._gam2_upper

    @gam2_upper.setter
    def gam2_upper(self, value):
        if type(value) is not float and type(value) is not int:
            raise TypeError("'gam2_upper' must be a float or a int")
        if value <= 0:
            raise ValueError("'gam2_upper' mustbe strictly positive")
        self._gam2_upper = value

    @property
    def mu_bar(self):
        return self._mu_bar

    @mu_bar.setter
    def mu_bar(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'mu_bar' must be a Numpy array")
        if value.ndim != 1 or np.squeeze(value).ndim != 1:
            raise ValueError("'mu_bar' must have only one dimension")
        if value.shape[0] != self.nc:
            raise ValueError("'mu_bar' must be of size 'nc'")
        self._mu_bar = value

    @property
    def eps_mu(self):
        return self._eps_mu

    @eps_mu.setter
    def eps_mu(self, value):
        if type(value) is not np.ndarray:
            raise TypeError("'self.eps_mu' must be a Numpy array")
        if value.ndim != 1 or np.squeeze(value).ndim != 1:
            raise ValueError("'self.eps_mu' must have only one dimension")
        if value.shape[0] != self.nc:
            raise ValueError("'self.eps_mu' must be of size 'nc'")
        self._eps_mu = value

    @property
    def lbd_mu(self):
        return self._lbd_mu

    @lbd_mu.setter
    def lbd_mu(self, value):
        if value is None:
            self._lbd_mu = value
        else:
            if type(value) is not np.ndarray:
                raise TypeError("'lbd_mu' must be a Numpy array")
            if value.ndim != 2 or np.squeeze(value).ndim != 2:
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
        if value.ndim != 1 or np.squeeze(value).ndim != 1:
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
        if value.ndim != 1 or np.squeeze(value).ndim != 1:
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
        if value.ndim != 1 or np.squeeze(value).ndim != 1:
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
            self._lbd_sig = value
        else:
            if type(value) is not np.ndarray:
                raise TypeError("'lbd_sig' must be a Numpy array")
            if value.ndim != 2 or np.squeeze(value).ndim != 2:
                raise ValueError("'lbd_sig' must have two dimensions")
            if value.shape[0] != self.nc or value.shape[1] != self.nf:
                raise ValueError("'lbd_sig' must be of size ['nc', 'nf']")
            self._lbd_sig = value

    def _m1_constraints(self, w, eta, z, lower, upper):
        return [
            eta - (1 - z) * upper <= w,
            w <= eta - lower * (1 - z),
            lower * z <= w,
            w <= upper * z,
        ]

    def _m2_constraints(self, w, eta, z1, z2, lower, upper):
        return [
            w <= upper * z1,
            w <= upper * z2,
            w <= eta - lower * (1 - z1),
            w <= eta - lower * (1 - z2),
            w >= lower * (-1 + z1 + z2),
            w >= eta + upper * (-2 + z1 + z2),
            z1 <= 1,  # Maybe remove 4 lines
            z2 <= 1,
            lower <= eta,
            eta <= upper,
        ]

    # The following comment is for using confusing variable names like J, I,
    # etc. without being harassed by flake8.
    # flake8: noqa: E741
    def _cvn(self, flp: FLP):
        warnings.warn(
            "Deprecated inner function: don't need to convert variables",
            DeprecationWarning,
        )
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


class DRSolver(BASSolver):
    """
    A plain distributionally robust solver. It uses the `BASSolver` with the
    'lbds' hyperparameters to 0.

    Attributes
    ----------
    mu_bar : array-like[nc]
        Factor from decision variable `y` for computing the 'mean' of
        expected values of the uncertainty set.
    eps_mu : array-like[nc]
        Upper bound on the difference between expected values and the
        'mean' of expected values on the uncertainty set.
    sig_bar : array-like[nc]
        Factor from decision variable `y` for computing the 'mean' of
        variances of the uncertainty set.
    eps_lower_sig : array-like[nc]
        Lower bound on the second moments of the uncertainty set.
    eps_upper_sig : array-like[nc]
        Upper bound on the second moments of the uncertainty set.
    nf : int
        Number of possible locations for building facilities.
    nc : int
        Number of costumer sites.
    delt1_upper : float
        An upper bound on the dual variable delta_1. See [1].
    delt2_upper : float
        An upper bound on the dual variable delta_2. See [2].
    gam1_upper : float
        An upper bound on the dual variable gamma_1. See [1].
    gam2_upper : float
        An upper bound on the dual variable gamma_2. See [2].

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
        sig_bar: Optional[np.ndarray] = None,
        eps_lower_sig: Optional[np.ndarray] = None,
        eps_upper_sig: Optional[np.ndarray] = None,
        nf: Optional[int] = None,
        nc: Optional[int] = None,
        delt1_upper: Optional[float] = 100000,
        delt2_upper: Optional[float] = 100000,
        gam1_upper: Optional[float] = 100000,
        gam2_upper: Optional[float] = 100000,
    ):
        """
        Initiate a DRSolver object with respect to hyperparameters.

        Parameters
        ----------
        mu_bar : array-like[nc], optional
            Factor from decision variable `y` for computing the 'mean' of
            expected values of the uncertainty set. Default: Uniform over
            [20, 40).
        eps_mu : array-like[nc], optional
            Upper bound on the difference between expected values and the
            'mean' of expected values on the uncertainty set. Default: 0.
        sig_bar : array-like[nc], optional
            Factor from decision variable `y` for computing the 'mean' of
            variances of the uncertainty set. Default: Equals to `mu_bar`.
        eps_lower_sig : array-like[nc], optional
            Lower bound on the second moments of the uncertainty set.
            Default: 1.
        eps_upper_sig : array-like[nc], optional
            Upper bound on the second moments of the uncertainty set.
            Default: 1.
        nf : int, optional
            Number of possible locations for building facilities. Default: 10.
        nc : int, optional
            Number of costumer sites. Default: 20.
        """
        super(DRSolver, self).__init__(
            mu_bar=mu_bar,
            eps_mu=eps_mu,
            sig_bar=sig_bar,
            eps_lower_sig=eps_lower_sig,
            eps_upper_sig=eps_upper_sig,
            nf=nf,
            nc=nc,
            delt1_upper=delt1_upper,
            delt2_upper=delt2_upper,
            gam1_upper=gam1_upper,
            gam2_upper=gam2_upper,
        )
        self.lbd_mu = np.zeros((self.nc, self.nf))
        self.lbd_sig = np.zeros_like(self.lbd_mu)

    def solve(self, flp: FLP) -> bool:
        """
        Solve the problem `flp` and return a boolean depending on the success.
        """
        return super().solve(flp)
