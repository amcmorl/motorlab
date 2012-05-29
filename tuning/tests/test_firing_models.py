import numpy as np
from numpy.testing import assert_almost_equal, assert_
import tuning_change.firing_models as fm

def test_generate_ballistic_profiles():
    t,d,s,a = fm.generate_ballistic_profiles()
    assert t[0] == 0
    assert t[-1] == 1.
    assert d.max() == 1.
    
def test_generate_cell_properties():
    PD, PG = fm.generate_cell_properties()
    assert_almost_equal(np.sum(PD**2), 1.)

    PDk = [1., 0.3, 0.2]
    PDe = PDk / np.linalg.norm(PDk)
    assert np.allclose(PDe, PD)
    assert_almost_equal(np.sum(PG**2), 1.)
    
def test_calc_pos_and_dir_along_movement():
    time, disp, speed, acc = fm.generate_ballistic_profiles()
    start = np.array((0., 0., 0.))
    stop = np.array((1., 1., 1.))
    pos, drn = fm.calc_pos_and_dir_along_movement(disp, stop)

    # check start and end at the right place
    assert_almost_equal(pos[...,0], start)
    assert_almost_equal(pos[...,-1], stop)

def test_generate_firing_rate():
    time, disp, speed, acc = fm.generate_ballistic_profiles()
    #start = np.array((0., 0., 0.))
    stop = np.array((1., 1., 1.))
    pos, drn = fm.calc_pos_and_dir_along_movement(disp, stop)
    PD, PG = fm.generate_cell_properties()
    k = fm.generate_firing_rate(pos, drn, speed, acc, time, \
                                  PD, PG, model='k')
    assert_(np.all(k == 1.))
    
    ds = fm.generate_firing_rate(pos, drn, speed, acc, time, \
                                  PD, PG, model='ds')
    kds = fm.generate_firing_rate(pos, drn, speed, acc, time, \
                                  PD, PG, model='kds')
    assert_(np.allclose(ds + k, kds))
    assert_(kds.size == pos.shape[1] - 1)
    assert_(kds.argmax() == speed.argmax())
    
def test_do_tasks():
    '''
    Do movement end points equal task end points?
    '''
    targets = fm.targets_dict['co_oc_3d_26']
    PD = np.array([0.3, 0.3, 0.7])
    PD /= np.sqrt(np.sum(PD**2))
    PG = np.array([-0.6, -0.7, 0.3])
    PG /= np.sqrt(np.sum(PG**2))

    time, speed, frs, all_pos = \
        fm.do_tasks(targets=targets, model='kvp',
                    k={'k':2, 'p':0.5, 'd':0.4, 'n':35},
                    PD=PD, PG=PG, full_output=False)
    assert_(np.allclose(all_pos[:,-1,:], targets[:,3:]))

def test_do_tasks2():
    '''
    Do movement end points equal task end points?
    '''
    targets = fm.targets_dict['co_oc_3d_26']
    PD = np.array([0.3, 0.3, 0.7])
    PD /= np.sqrt(np.sum(PD**2))
    PG = np.array([-0.6, -0.7, 0.3])
    PG /= np.sqrt(np.sum(PG**2))
    
    time, speed, frs, all_pos = \
        fm.do_tasks(targets=targets, model='kvp',
                    k={'k':2, 'p':0.5, 'd':0.4, 'n':35},
                    PD=PD, PG=PG, full_output=False)
    assert_(np.allclose(all_pos[:,-1,:], targets[:,3:]))

    #mv = {'curvature' : 0., 'center' : 0.5, 'fwhm' : 0.2}
    #speed2, frs2, pos2, pds2, starts2, stops2 = tc_model.do_tasks2( \
    #    tasks=targets, model='kvp', mv=mv,
    #    PD=PD, PG=PG, n_bins=100, full_output=True)

    
