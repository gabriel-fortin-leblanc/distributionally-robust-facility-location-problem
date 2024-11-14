import numpy as np
import pytest

from ddro import pg


class TestFLP:

    @pytest.mark.parametrize(
        "nf, nc, sd, oc, tc, cf",
        [
            (
                2,
                2,
                np.array([1, 2]),
                np.array([10, 20]),
                np.array([[10, 20], [20, 30]]),
                np.array([1, 2]),
            )
        ],
    )
    def test_init(self, nf, nc, sd, oc, tc, cf):
        """No error in init"""
        print(cf.shape)
        flp = pg.FLP(nf, nc, sd, oc, tc, cf)
        assert flp.nf == nf
        assert flp.nc == nc
        assert np.array_equal(flp.sd, sd)
        assert np.array_equal(flp.oc, oc)
        assert np.array_equal(flp.tc, tc)
        assert np.array_equal(flp.cf, cf)
