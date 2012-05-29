from spherical import pol2cart, cart2pol
from numpy import exp, dot, log, radians, diff
#from numpy.random import standard_normal as normal
from numpy.random import poisson
from motorlab.datasets import datasets
from motorlab.datacollection import DataCollection
from motorlab.tuning.calculate import kinematics
from motorlab.tuning.fit import glm_any_model
from vectors import unitvec
from numpy.testing import assert_array_less

def test_glm_any_model():
    nbin = 10
    align = 'hold'
    lag = 0.1 # s

    ds = datasets['small']
    dc = DataCollection(ds.get_files())
    dc.add_unit(ds.get_units()[0], lag)
    bnd = dc.make_binned(nbin=nbin, align=align)
    ntask, nrep, nunit, nbin = bnd.shape

    # make perfect test counts
    # direction-only model
    tp = radians([20, 10]) # degrees
    b0 = log(10) # log(Hz)
    pd = pol2cart(tp)

    drn = kinematics.get_idir(bnd.pos, axis=2)
    rate = exp(dot(drn, pd) + b0)
    window_size = diff(bnd.bin_edges, axis=2)
    count_mean = rate * window_size
    count = poisson(count_mean)
    count = count[:,:,None]
    bnd.set_PSTHs(count)

    # now fit data for pd
    count, pos, time = bnd.get_for_regress()
    bnom, bse_nom = glm_any_model(count, pos, time, model='kd')
    pd_exp = unitvec(bnom['d'])
    tp_exp = cart2pol(pd_exp)
    
    acceptable_err = 0.05 # about 3 degrees absolute error
    err = abs(tp - tp_exp)
    assert_array_less(err, acceptable_err)
