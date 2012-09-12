import numpy as np
import motorlab.tuning.calculate.pds as cpd
from proc_tools import edge2cen
import spherical_stats as sphere_stat
#import motorlab.tuning.firing_models as tc_model

def test_average_edges_to_centers_nd():
    test = np.random.randint(0, 10, size=(5,4,3))

    res = edge2cen(test, axis=0)
    assert(res.shape == (4,4,3))
    assert(res[0,0,0] == (test[0,0,0] + test[1,0,0]) / 2.)

    res = edge2cen(test, axis=1)
    assert(res.shape == (5,3,3))
    assert(res[0,0,0] == (test[0,0,0] + test[0,1,0]) / 2.)

    res = edge2cen(test, axis=2)
    assert(res.shape == (5,4,2))
    assert(res[0,0,0] == (test[0,0,0] + test[0,0,1]) / 2.)

def test_bootstrap_pd_stats():
    '''
    Notes
    -----
    create a population of 3d data points
    with a known mean and certain spread

    take a sample of n points from that population
    get the mean and std error of that sample

    use bootstrap_pd_stats to get kappa
    95% of time, real mean pd should lie within kappa of sample means
    '''
    n_trials = 100
    pop_size = 1e6
    smp_size = 40
    alpha = 0.05 # used to be 0.95 here
    pds = [[0., 0., 1.], [0.,1.,0.], [1.,0.,0.]]
    sss = [[0.1, 0.1, 0.1], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]

    for pd in pds:
        for ss in sss:
            print "--------------------"
            print "pd: ", pd
            print "ss: ", ss
            failures = 0
            for i in range(n_trials):
                pop = np.random.standard_normal(size=(pop_size,3)) * ss + pd
                smp_inds = np.random.randint(pop_size, size=smp_size)
                smp = pop[smp_inds]
                smp_mean = np.mean(smp, axis=0)
                smp_k = np.sqrt(np.sum(smp_mean**2))
                smp_pd = smp_mean / smp_k
                smp_stderr = np.std(smp, axis=0, ddof=1) / np.sqrt(smp_size)
                
                # estimate k_s, kappa using bootstrap_pd_stats
                k_s, kappa, R = cpd.bootstrap_pd_stats( \
                    smp_pd, smp_stderr, smp_k, n_samp=smp_size)
                interangle = np.arccos(np.dot(pd, smp_pd))
                theta = sphere_stat.measure_percentile_angle_ex_kappa( \
                    kappa)
                print "%2d- interangle: %6.4f theta %6.4f" % \
                    (i, interangle, theta),
                if interangle > theta:
                    failures += 1
                    print '*'
                else:
                    print '-'
            print "failures: %.2f" % (failures / float(n_trials))
            assert failures < (alpha * n_trials)

# def test_calc_vel_same_points():
#     speed, mid_speed = tc_model.generate_speed_profile()
#     tasks = tc_model.targets_dict['co_oc_3d_26']
#     pos = tc_model.calc_pos_along_movement(speed, tasks,
#                                            curvature=1.0)
#     vel = tc_calc.calc_vel_same_pts(pos)
#     speed2 = np.sqrt(np.sum(vel**2, axis=2))
#     #assert(np.all(speed == speed2))
#     return speed, speed2

