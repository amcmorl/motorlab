import numpy as np
import rpy2.robjects as ro
import matplotlib as mpl
import matplotlib.pyplot as plt

ro.r('source("fit_noCV.R")')
ro.r('source("simulate_realkinematics.R")')
fit_noCV = ro.r('fit_noCV')

data = ro.r('simu.realpos(sd.factor=0.0001)')

out = fit_noCV(data)

sse = np.array(out.r['sse'][0])

# plot out to a postscript file
super_star = "$^*$"
modellist = ["kd","kdp","kds","kdps","kv","kvp","kvs","kvps"]
modellist += [model + super_star for model in modellist]
#"kd*","kdp*","kds*","kdps*","kv*","kvp$^*$","kvs*","kvps*"]
fig = plt.figure(figsize = (4,3))
l, b = 0.15, 0.225
r, t = 0.00, 0.05
w, h = 1 - l - r, 1 - t - b
ax = fig.add_axes([l, b, w, h])
ax.imshow(np.log(sse), origin='lower', cmap=mpl.cm.Greys_r)
axes = [ax.xaxis, ax.yaxis]
rots = ['vertical', 'horizontal']
for axis, rot in zip(axes, rots):
    axis.set_ticks(range(0, len(modellist)))
    axis.set_ticklabels(modellist, rotation=rot, fontsize='small')
    axis.set_ticks_position('none')

