import numpy as np

from motorlab.binned_data import _estimate_count_from_rate, \
    _make_count_from_rate

def test_estimate_count_from_rate():
    rate = np.array([-1, 0, 1, 2, 3, np.nan])
    bin_edges = np.arange(rate.size + 1) * 0.5
    have = _estimate_count_from_rate(rate, bin_edges)
    want = np.array([-0.5, 0., 0.5, 1.0, 1.5, np.nan])
    np.testing.assert_array_almost_equal(have, want)
    
def test_make_count_from_rate():
    rate = np.array([-1, 0, 1, 2, 3, np.nan])
    bin_edges = np.arange(rate.size + 1) * 0.5
    
    nrep = 1e6
    rate = np.tile(rate, (nrep,1))
    bin_edges = np.tile(bin_edges, (nrep,1))
    
    have = _make_count_from_rate(rate, bin_edges)
    have_ = np.mean(have, axis=0)
    want = np.array([0, 0, .5, 1, 1.5, np.nan])
    
    # hopefully this is enough repeats to average out noise reliably
    np.testing.assert_array_almost_equal(have_, want, decimal=2)
