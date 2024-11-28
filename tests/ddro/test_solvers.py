import numpy as np
import pytest

from ddro import pg, solvers


class PSolver:

    @pytest.mark.slow
    def test_solve(self):
        flp = pg.flp_generator()
        solver = solvers.PSolver()
        assert solver.solve(flp)
        assert type(solver.y) is np.ndarray


class TestBASSolver:

    def test_init_default(self):
        """No error in init"""
        solver = solvers.BASSolver()
        assert (solver.mu_bar <= 40).all()
        assert (solver.mu_bar >= 20).all()
        assert (solver.eps_mu == 0).all()
        assert solver.lbd_mu is None
        assert np.array_equal(solver.sig_bar, solver.mu_bar)
        assert (solver.eps_lower_sig == 1).all()
        assert (solver.eps_upper_sig == 1).all()
        assert solver.lbd_sig is None
        assert solver.nf == 10
        assert solver.nc == 20

    @pytest.mark.slow
    def test_solve(self):
        flp = pg.flp_generator()
        solver = solvers.BASSolver()
        assert solver.solve(flp)
        assert type(solver.y) is np.ndarray


class TestDRSolver:

    def test_init_default(self):
        """No error in init"""
        solver = solvers.DRSolver()
        assert (solver.mu_bar <= 40).all()
        assert (solver.mu_bar >= 20).all()
        assert (solver.eps_mu == 0).all()
        assert np.array_equal(solver.sig_bar, solver.mu_bar)
        assert (solver.eps_lower_sig == 1).all()
        assert (solver.eps_upper_sig == 1).all()
        assert solver.nf == 10
        assert solver.nc == 20

    @pytest.mark.slow
    def test_solve(self):
        flp = pg.flp_generator()
        solver = solvers.DRSolver()
        assert solver.solve(flp)
        assert type(solver.y) is np.ndarray
