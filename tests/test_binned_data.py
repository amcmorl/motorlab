import numpy as np

from motorlab.binned_data import _estimate_count_from_rate, \
    _make_count_from_rate, BinnedData

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

def test_with_only():
    # build a bnd where only shape of attributes matters
    ntask, nrep, nunit, nbin, ndim = 3,2,2,5,3
    
    bin_edges  = np.zeros([ntask, nrep, nbin + 1])
    pos        = np.zeros([ntask, nrep, nbin + 1, ndim])
    count      = np.zeros([ntask, nrep, nunit, nbin], dtype=int)
    tasks      = np.zeros([ntask, ndim * 2])
    unit_names = np.array(['u%d' % (x) for x in range(nunit)])
    align      = 'hold'
    lags       = np.zeros([nunit])
    bnd = BinnedData(bin_edges, pos, tasks, unit_names, align, \
                     count=count, lags=lags)
    
    # one dset only
    new = bnd.with_only(1, keep_dims=False)
    np.testing.assert_equal(new.count.shape, [ntask, nrep, nbin])
    np.testing.assert_equal(new.unit_names.shape, ())
    np.testing.assert_equal(new.lags.shape, ())
    
    # one dset, preserving dimensionality
    new = bnd.with_only(1, keep_dims=True)
    np.testing.assert_equal(new.count.shape, [ntask, nrep, 1, nbin])
    np.testing.assert_equal(new.unit_names.shape, (1,))
    np.testing.assert_equal(new.lags.shape, (1,))
    
    # one dset, specified as array
    new = bnd.with_only(np.array([1]), keep_dims=False)
    np.testing.assert_equal(new.count.shape, [ntask, nrep, 1, nbin])
    np.testing.assert_equal(new.unit_names.shape, (1,))
    np.testing.assert_equal(new.lags.shape, (1,))
    
