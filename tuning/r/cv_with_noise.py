import numpy as np
import rpy2.robjects as ro
import matplotlib as mpl
import matplotlib.pyplot as plt

ro.r('source("fit_CV.R")')
ro.r('source("simulate_realkinematics.R")')
fit_CV = ro.r('fit_CV')

noise = 0.0001
data = ro.r('simu.realpos(sd.factor=%0.4f)' % noise)
out = fit_CV(data, 10)
out_ar = np.array([np.array(x) for x in list(out)])
n_mods = out_ar.shape[0]
#out_ar.tofile('cv_matrix_with_%0.0enoise.txt' % noise, sep=' ')

# rejig out_ar to get repeats in correct order for plotting
# just use 9 repeats, since is sqrt-able
out_jig = out_ar[:,:-1,:].reshape(n_mods,3,3,n_mods).transpose([0,2,3,1]).reshape(16*3,16*3)
save_dir = '../../tuningpaper/figures/fig_method/'

# plot matrix with uncertainty in squares
super_star = "$^*$"
modellist = ["kd","kdp","kds","kdps","kv","kvp","kvs","kvps"]
modellist += [model + super_star for model in modellist]
#"kd*","kdp*","kds*","kdps*","kv*","kvp$^*$","kvs*","kvps*"]
fig = plt.figure(figsize = (4,3))
l, b = 0.15, 0.225
r, t = 0.00, 0.05
w, h = 1 - l - r, 1 - t - b
ax = fig.add_axes([l, b, w, h])
ax.imshow(np.log(out_jig), origin='lower', cmap=mpl.cm.Greys_r)
axes = [ax.xaxis, ax.yaxis]
rots = ['vertical', 'horizontal']
for axis, rot in zip(axes, rots):
    axis.set_ticks(np.arange(len(modellist)) * 3 + 2)
    axis.set_ticklabels(modellist, rotation=rot, fontsize='small')
    axis.set_ticks_position('none')

fig.savefig(save_dir + 'cv_matrix_%0.0e_noise.pdf' % noise)
# low noise generates same as no_confusion_gam.py - good

# pick kvs=6, or kvps=7
roi_name = 'kvs'
rois = {'kvps' : 7,
        'kvs' : 6}
roi = out_ar[rois[roi_name]]

# plot of CV repeats
fig2 = plt.figure(figsize=(4,3))
l, b = 0.2, 0.225
r, t = 0.05, 0.05
w, h = 1 - l - r, 1 - t - b
ax2 = fig2.add_axes([l, b, w, h])
ax2.imshow(np.log(roi), cmap=mpl.cm.Greys_r, origin='lower')
ax2.set_xticks(np.arange(n_mods) + 0.5)
ax2.set_xticklabels(modellist, rotation='vertical', fontsize='small')
ax2.set_yticks(np.arange(roi.shape[0]))
ax2.xaxis.set_ticks_position('none')
ax2.yaxis.set_ticks_position('none')
fig2.savefig(save_dir + roi_name + '_cv_repeats_img_%0.0e_noise.pdf' % noise)

# log plot of means and ses for roi
means = roi.mean(axis=0)
errs = roi.std(axis=0) / np.sqrt(roi.shape[0])
lnperr = np.log(means + errs) - np.log(means)
lnmerr = -np.log(means - errs) + np.log(means)
lnerr = np.vstack((lnperr, lnmerr))
fig3 = plt.figure(figsize=(4,3))
l, b = 0.2, 0.225
r, t = 0.05, 0.05
w, h = 1 - l - r, 1 - t - b
ax3 = fig3.add_axes([l, b, w, h])
ax3.bar(np.arange(n_mods), -np.log(means), color='#B0B0B0')
ax3.errorbar(np.arange(n_mods) + 0.4, -np.log(means), lnerr,
             fmt=None, ecolor='k')
for loc, spine in ax3.spines.iteritems():
    if loc == 'left':
        spine.set_position(('outward', 5))
    elif loc == 'bottom':
        spine.set_position(('data', 0.0))
    else:
        spine.set_color('none')
ax3.set_ylabel('-log(SSE)')
ax3.set_xticks(np.arange(n_mods) + 0.5)
ax3.set_xticklabels(modellist, rotation='vertical', fontsize='small')
ax3.xaxis.set_ticks_position('none')
ax3.yaxis.set_ticks_position('left')
for tick in ax3.yaxis.get_ticklines():
    tick.set_markeredgewidth(1.0)

fig3.savefig(save_dir + roi_name + '_cv_repeats_bar_%0.0e_noise.pdf' % noise)
plt.show()

# noise levels:
#
# 1e-4 - can definitely distinguish all
# 5e-4
# 7.5e-4
# 1e-3

