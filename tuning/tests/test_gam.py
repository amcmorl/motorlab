from motorlab.datasets import datasets
from motorlab.datacollection import DataCollection
from motorlab.tuning import gam

def test_gam_runs_on_simple():
    ds = datasets['small']
    unit = ds.get_units()[0]

    threshold = 5. # Hz
    lag = 0.1 # seconds
    nbin = 10 # bins
    align = 'hold'
    ncv = 4

    dc = DataCollection(ds.get_files())
    dc.add_unit(unit, lag)
    bnd = dc.make_binned(nbin=nbin, align=align, do_rate=True)
    # need to change this to do_rate=False once running on counts is fixed

    data = gam.format_data_wrap(bnd.unbiased_rate, bnd.bin_edges, bnd.pos, drop=True)
    gamres = gam.GAMResult(gam.fit_CV(data, 4, 50))

def test_gam_runs_on_complex():
    ds = datasets['frank-osmd']
    unit = ds.get_units()[0]

    threshold = 5. # Hz
    lag = 0.1 # seconds
    nbin = 10 # bins
    align = 'hold'
    ncv = 4

    dc = DataCollection(ds.get_files())
    dc.add_unit(unit, lag)
    bnd = dc.make_binned(nbin=nbin, align=align, do_rate=True)
    # need to change this to do_rate=False once running on counts is fixed

    data = gam.format_data_wrap(bnd.unbiased_rate, bnd.bin_edges, bnd.pos, drop=True)
    gamres = gam.GAMResult(gam.fit_CV(data, 4, 50))
