import numpy as np
import rpy2.robjects as ro
import matplotlib as mpl
import matplotlib.pyplot as plt
import tuning_change.simulate as tc_model
import sys
sys.path.append('/home/amcmorl/files/pitt/tuning_change/code/batch_files/')
import load
from tuning_change import gam

from IPython.kernel import client
mec = client.MultiEngineClient()
assert len(mec.get_ids() >= 2)

class Options:
    stage = 3
    data = 'sim'
    plots = ['mse', 'comparison']
    k = {'k' : 2., 'p' : 1., 'd' : 1., 'v' : 1., 'n' : 35}
    n_cv = 4
    noise = 0.0001
    n_bins = 10
    align = 'speed'
    unit_number = 0
    PD = np.array([0.3, 0.3, 0.7])
    PG = np.array([-0.6, -0.7, 0.3])
    coi = {'dset' : 0, 'trial' : 0}
    models = gam.modellist[:2]

    def __init__(self):
        assert np.sqrt(self.n_cv) % 1 == 0
        self.PD /= np.linalg.norm(self.PD)
        self.PG /= np.linalg.norm(self.PG)
        self.n_mods_out = len(self.modellist)

o = Options()
        
if o.stage >= 1:
    cell = load.load_cell(unit_number=o.unit_number)
    srt = cell.construct_PSTHs2(n_bins=o.n_bins, align=o.align)
    n_trials, n_reps, n_bins, n_dims = srt.pos.shape
    # pos /= np.nanmax(pos) # still don't know why I did this, so I won't
    if o.data == 'sim':
        print "Using simulated data..."
        #del PSTHs, asb, aeb, ss
        n_mods_in = len(o.models)
        frs = np.empty((n_trials, n_reps, n_mods_in, n_bins))
        
        for i_model, gen_model in enumerate(o.models):
            for i_trial in xrange(n_trials):
                for i_rep in xrange(n_reps):
                    frs[i_trial, i_rep, i_model, :] = \
                        tc_model.generate_firing_rate2( \
                        srt.pos[i_trial, i_rep], o.PD, o.PG, k=o.k,
                        model=gen_model)

        # add noise to data same way as Zhanwu does
        frs = np.random.normal(loc=0, scale=o.noise * np.sqrt(frs),
                               size=frs.shape) + frs
        drop = False
    elif o.data == 'real':
        print "Using real data..."
        n_reps = PSTHs.shape[1]
        frs = PSTHs[..., None, :]
        n_mods_in = frs.shape[2]
        drop = True
    else:
        raise Exception('data option not known')
    print "Firing rate shape:", frs.shape

if o.stage >= 2:
    data = gam.format_data_wrap(frs, srt.bin_edges, srt.pos, drop=drop)
    out = gam.GAMResult(gam.fit_CV(data, o.n_cv))
    #n_mods_in, n_mods_out = mse_matrix.shape[0], mse_matrix.shape[-1]
    # mse_matrix.tofile('cv_matrix_with_%0.0enoise_unit115_1_kv_notX.txt' %
    #                   o.noise, sep=' ')


if 'mse' in o.plots and (o.stage >= 3):
    if n_mods_in > 1:
        gam.plot_cvs_matrix(out.mse, o.models)
#         s = np.sqrt(o.n_cv)
#         n_mods_out = mse_matrix.shape[-1]
#         msem_jig = np.transpose( \
#             mse_matrix[:,:,:].reshape(n_mods_in, s, s, n_mods_out), \
#                 [0,2,3,1]).reshape(n_mods_in*s, n_mods_out*s)
#         # rejig out_ar to get repeats in correct order for plotting
#         # just use max sqrtable number smaller than n_cv
        
#         # plot matrix with uncertainty in squares
#         #modellist = ["kd","kdp","kds","kdps","kv","kvp","kvs","kvps"]
#         #modellist += [model + super_star for model in modellist]
#         fig = plt.figure(figsize = (4,3))
#         l, b = 0.15, 0.225
#         r, t = 0.05, 0.05
#         w, h = 1 - l - r, 1 - t - b
#         ax = fig.add_axes([l, b, w, h])
#         ax.imshow(np.log(msem_jig), origin='lower', cmap=mpl.cm.Greys_r)
#         axes = [ax.xaxis, ax.yaxis]
#         ax.xaxis.set_ticks(np.arange(o.n_mods_out) * s + s/2.)
#         ax.yaxis.set_ticks(np.arange(n_mods_in) * s + s/2.)
#         ax.xaxis.set_ticklabels(o.modellist, rotation='vertical', \
#                                     fontsize='small')
#         ax.yaxis.set_ticklabels(o.models, rotation='horizontal', \
#                                     fontsize='small')
#         ax.xaxis.set_ticks_position('none')
#         ax.yaxis.set_ticks_position('none')
#         fig.savefig(cell.uniq_name() + '_cv_mse_matrix.png')
    else: # only one dataset
        fig = plt.figure()
        l, b = 0.15, 0.225
        r, t = 0.05, 0.05
        w, h = 1 - l - r, 1 - t - b
        ax = fig.add_axes([l, b, w, h])
        ax.plot(-np.log(mse_matrix.T), drawstyle='steps-mid')
        super_star = "$^*$"
        modellist = ["kd","kdp","kds","kdps","kv","kvp","kvs","kvps"]
        modellist += [model + super_star for model in modellist]
        ax.set_xticks(range(0, len(modellist)))
        ax.set_xticklabels(modellist, rotation='vertical')
        fig.savefig(cell.uniq_name() + '_cv_mse_profile.png')

if 'comparison' in o.plots and (o.stage >= 3):

    gam.plot_prediction_comparison(out.pred, out.actual, out.trials, n_reps)
    
#     tids = out.trials / n_reps # converts trial ids to task numbers
#     act_av, pred_av = gam.average_cvs(out.actual, out.pred, tids)
    
#     n_cols = n_trials / 13
#     n_rows = n_trials / n_cols
#     fig = plt.figure(figsize = (16,11))
#     axes = []
#     for i_trial in xrange(n_trials):
#         col = i_trial / 13
#         row = i_trial % 13

#         lmarg, rmarg, hmarg = 0.05, 0.05, 0.01
#         tmarg, bmarg, vmarg = 0.05, 0.05, 0.01
#         w = (1 - lmarg - rmarg - (n_cols - 1) * hmarg) / n_cols
#         h = (1 - tmarg - bmarg - (n_rows - 1) * vmarg) / n_rows
#         l = lmarg + col * (w + hmarg)
#         b = bmarg + row * (h + vmarg)
#         ax = fig.add_axes([l, b, w, h])
        
#         ax.plot(act_av[o.coi['dset'], i_trial], 'ko--', markersize=4)
#         ax.plot(pred_av[o.coi['dset'],:, i_trial].T)
#         if not row == 0: # not row 0
#             ax.xaxis.set_ticks([])
#             for loc, spine in ax.spines.iteritems():
#                 if loc in ['bottom', 'top']:
#                     spine.set_color('none')
#         else: # row 0
#             for loc, spine in ax.spines.iteritems():
#                 if loc == 'top':
#                     spine.set_color('none')
#                 elif loc == 'bottom':
#                     spine.set_position(('outward', 5))
#             ax.xaxis.set_ticks_position('bottom')
#         if not col == 0:
#             ax.yaxis.set_ticks([])
#             for loc, spine in ax.spines.iteritems():
#                 if loc in ['left', 'right']:
#                     spine.set_color('none')
#         else: # col 0
#             for loc, spine in ax.spines.iteritems():
#                 if loc == 'right':
#                     spine.set_color('none')
#                 elif loc == 'left':
#                     spine.set_position(('outward', 5))
#             ax.yaxis.set_ticks_position('left')
#         axes.append(ax)
    #fig.savefig(cell.uniq_name() + '_act_vs_preds_52tasks.png')
                
if len(o.plots) > 0:
    plt.show()
    
