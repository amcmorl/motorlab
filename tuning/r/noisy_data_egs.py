import numpy as np
import rpy2.robjects as ro
import matplotlib as mpl
from matplotlib.pyplot import figure, show
import tuning_change.display as tc_disp

opts = {'full'      : False,
        'sub_slice' : slice(25,40),
        'noises'    : [0.0001, 0.0005, 0.005],
        'noise_int' : 1,
        'mvmt'      : [slice(  0,  98), slice( 98, 188),
                       slice(188, 275), slice(275, 358),
                       slice(358, 450), slice(450, 540),
                       slice(540, 630), slice(630, None)],
        'plot_mvmts' : [5,7],
        'plot_symbols' : ['-', '--'],
        'rois'     : {'kd'  : 0, 'kdp'  : 1,
                      'kds' : 2, 'kdsp' : 3,
                      'kv'  : 4, 'kvp'  : 5,
                      'kvs' : 6, 'kvsp' : 7},
        'roi_name' : 'kvsp',
        'plot_pos' : False,
        'save_dir' : '../../paper/figures/fig_method/'}

noise = opts['noises'][opts['noise_int']]

ro.r('source("fit_CV.R")')
ro.r('source("simulate_realkinematics.R")')

d = {}
dname = '%0.0e' % noise
d[dname] = ro.r('simu.realpos(sd.factor=%0.4f)' % noise)
fr  = np.array(d[dname].r['firing.rate'][0]) # shape 714,16
fr /= fr.max()
kin = np.array(d[dname].r['Xmat'][0]) # shape 714,12
pos = kin[:,4:7]

fig = figure(figsize=(4,3)) # firing rate
axs = []
l, b = 0.2, 0.175
r, t = 0.05, 0.05
w, h = 1 - l - r, 1 - t - b
fax = fig.add_axes([l, b, w, h])

xyz_axs = None
if opts['full']:
    sl=slice(None,None)
else:
    sl=opts['sub_slice']
for i, sym in zip(opts['plot_mvmts'], opts['plot_symbols']):
    if opts['plot_pos']:
        xyz_axs = tc_disp.plot_xyz(pos[opts['mvmt'][i]], figsize=(4,3),
                                   axes=xyz_axs, color='k', linestyle=sym)
    fxs = np.linspace(0, 1., fr[opts['mvmt'][i]].shape[0])
    fax.plot(fxs[sl], fr[opts['mvmt'][i], opts['rois'][opts['roi_name']]][sl],
             'k%s' % sym) # 2nd col: 6 = kvs, 7 = kvsp
if opts['plot_pos']:
    tc_disp.scale_axes(xyz_axs, ymin='auto')

for loc, spine in fax.spines.iteritems():
    if loc in ['left', 'bottom']:
        spine.set_position(('outward', 5))
    elif loc in ['right', 'top']:
        spine.set_color('none')
    else:
        raise ValueError('unknown spine location: %s' % loc)
fax.xaxis.set_ticks_position('bottom')
for tick in fax.xaxis.get_ticklines():
    tick.set_markeredgewidth(1.0)
fax.yaxis.set_ticks_position('left')
for tick in fax.yaxis.get_ticklines():
    tick.set_markeredgewidth(1.0)

if opts['full']:
    fax.set_xticks([0, 1])
    fax.set_yticks([0, 1])
fax.set_xlabel('time')
fax.set_ylabel('firing rate (f/f$_{max}$)')

show()
if opts['full']:
    cu = ''
else:
    cu = 'closeup_'
fig.savefig(opts['save_dir'] + opts['roi_name'] + '_fr_%s%0.0e.pdf' % \
                (cu, noise))
if opts['plot_pos']:
    xyz_fig = xyz_axs[0].figure
    xyz_fig.savefig(save_dir + 'position_egs.pdf' % noise)
