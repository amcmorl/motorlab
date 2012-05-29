import numpy as np
import motorlab.tuning_change.gaussmix.simulate as gsim
import motorlab.tuning_change.gaussmix.fit_amp as gfit_amp
import motorlab.tuning_change.gaussmix.fit_pd as gfit_pd
import motorlab.tuning_change.gaussmix as gmix
import motorlab.tuning_change.datacollection as dcol
import curves

from numpy.testing import assert_, assert_equal, assert_array_almost_equal

def test_fit_amp_sim_compatibility():
    options = gsim.Options()
    bnd = gsim.get_start_srt(options)
    
    # test if gfit_amp.gsmx with pset gives same result
    # as gsim.make_template with amp and pd
    bin = np.linspace(0, options.tt, options.nbin + 1)
    theta = gmix.expand_theta(gmix.calc_theta_target(bnd.parent.tasks,
                                                     options.pd), bnd)
    res0 = gsim.make_template(bin, options.amp, options.width,
                              options.center, theta)
    pset = gfit_amp.make_p_from_options(bnd.parent.tasks, options)
    b0, amp, width, center = gfit_amp.unpackp(pset, options.n)
    res1 = gfit_amp.gsmx(bin, amp, width, center)
    res1b = np.tile(res1[:,None,:], (1, bnd.PSTHs.shape[1], 1))
    assert_array_almost_equal(res0, res1b)

def test_fit_pd_sim_compatibility():
    options = gsim.Options()
    bnd = gsim.get_start_srt(options)

    # test if gfit_amp.gsmx with pset gives same result
    # as gsim.make_template with amp and pd
    bin = np.linspace(0, options.tt, options.nbin + 1)
    theta = gmix.expand_theta(gmix.calc_theta_target(bnd.parent.tasks,
                                                     options.pd), bnd)
    res0 = gsim.make_template(bin, options.amp, options.width,
                              options.center, theta)
    pset = gfit_pd.make_p_from_options(bnd, options)
    b0, pd, amp, width, center = gfit_pd.unpackp(pset, options.n)
    res1 = gfit_pd.gsmx(bin, bnd.parent.tasks, pd, amp, width, center)
    res1b = np.tile(res1[:,None,:], (1, bnd.PSTHs.shape[1], 1))
    assert_array_almost_equal(res0, res1b)

def test_make_p0_pd_appropriate():
    options = gsim.Options()
    #bnd = gsim.get_start_srt(options)
    p0 = gfit_pd.make_p0(options)

    # correct length
    assert_equal(p0.size, 11)

    # check baseline is in range 0:20 Hz, which seems reasonable
    assert_((p0[0] >= 0) & (p0[0] <= 20.)) 

    # check thetas are in range 0:pi, phis in 0:2pi
    thetas = p0[np.array([1,3])]
    assert_(np.all(thetas >= 0.))
    assert_(np.all(thetas < np.pi))
    phi = p0[np.array([2,4])]
    assert_(np.all(phi >= 0.))
    assert_(np.all(phi < (2 * np.pi)))
    
    # check amplitudes are in range 10:150 Hz
    assert_(np.all(p0[5:7] >= 0) & np.all(p0[5:7] <= 150.))
    assert_(np.all(p0[5:7] >= 0) & np.all(p0[5:7] <= 150.))

    # check widths are in range 0:0.5 s
    assert_(np.all(p0[-4:-2] >= 0) & np.all(p0[-4:-2] <= .5))
    assert_(np.all(p0[-4:-2] >= 0) & np.all(p0[-4:-2] <= .5))
    
    # check centers are in range 0:1 s
    assert_(np.all(p0[-2:] >= 0) & np.all(p0[-2:] <= 1))
    assert_(np.all(p0[-2:] >= 0) & np.all(p0[-2:] <= 1))

def test_make_p0_amp_appropriate():
    options = gsim.Options()
    bnd = gsim.get_start_srt(options)
    p0 = gfit_amp.make_p0(bnd, options)

    # correct length
    ntask = bnd.PSTHs.shape[0]
    assert_equal(p0.size, options.n * (ntask + 2) + 1)

    # check baseline is in range 0:20 Hz, which seems reasonable
    assert_((p0[0] >= 0) & (p0[0] <= 20.)) 
    
    # check amplitudes are in range 10:150 Hz
    assert_(np.all(p0[1:ntask * options.n + 1] >= 0) &
                       np.all(p0[1:ntask * options.n + 1] <= 150.))
    assert_(np.all(p0[1:ntask * options.n + 1] >= 0) &
                       np.all(p0[1:ntask * options.n + 1] <= 150.))

    # check widths are in range 0:0.5 s
    assert_(np.all(p0[-4:-2] >= 0) & np.all(p0[-4:-2] <= .5))
    assert_(np.all(p0[-4:-2] >= 0) & np.all(p0[-4:-2] <= .5))
    
    # check centers are in range 0:1 s
    assert_(np.all(p0[-2:] >= 0) & np.all(p0[-2:] <= 1))
    assert_(np.all(p0[-2:] >= 0) & np.all(p0[-2:] <= 1))

def test_options():
    options = gsim.Options()
    n = options.n
    assert_(np.isscalar(n))
    assert_equal(options.pd.shape, [n,3])
    
def test_pd_make_p_from_options():
    options = gsim.Options()
    pset = gfit_pd.make_p_from_options(options)
    b0, pd, amp, width, center = gfit_pd.unpackp(pset, options.n)
    assert_equal(pd.shape, [options.n,3])
    assert_equal(b0, options.b0)
    assert_array_almost_equal(pd, options.pd)
    assert_equal(amp, options.amp)
    assert_equal(width, options.width)
    assert_equal(center, options.center)
    
def test_pd_unpackp():
    options = gsim.Options()
    p0 = gfit_pd.make_p0(options)
    b0, pd, amp, width, center = gfit_pd.unpackp(p0, options.n)
    assert_equal(b0, 5)
    assert_array_almost_equal(pd, np.array([[0., 0., 1.], [0., 0., 1.]]))
    assert_equal(amp, [75, 75])
    assert_equal(width, curves.fwhm2k(.3))
    assert_equal(center, [options.tt/3., options.tt * 2 / 3.])

def test_amp_make_p_from_options():
    options = gsim.Options()
    dc = dcol.DataCollection(options.files)
    pset = gfit_amp.make_p_from_options(dc.tasks, options)
    ntask = dc.tasks.shape[0]
    assert_equal(pset.shape, (options.n * (ntask + 2) + 1,))
    b0, amp, width, center = gfit_amp.unpackp(pset, options.n)
    assert_equal(b0, options.b0)
    assert_equal(width, options.width)
    assert_equal(center, options.center)
