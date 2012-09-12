import motorlab.tuning_change.simulate.rates as simrates
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from curves import squash

def test_make_tgt_dir_rate():
    tgt = np.array([[1,1,1]])
    pos = np.tile(np.linspace(0, 1, 10, endpoint=False)[None, None, :, None],
                  (1, 1, 1, 3))
    pd = np.array([0, 0, 1])[None] # shape (nunit, 3)
    kd = np.array([1.])
    irt3 = 1 / np.sqrt(3.)
    sqp = {'m' : irt3,
           'a' : 2,
           'b' : 0.1}
    rate = simrates.make_tgt_dir_rate(tgt, pos, pd, kd, sqp=sqp)
    assert_array_almost_equal(rate, squash(irt3, sqp))
