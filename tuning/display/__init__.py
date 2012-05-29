import numpy as np

def print_stats(rates, name):
    print '-------%s--------' % name
    if np.rank(rates) == 2:
        n_dirs, n_bins = rates.shape
        n_reps = 1
    else:
        n_dirs, n_reps, n_bins = rates.shape
    print 'directions: %d, repeats: %d, bins: %d' % (n_dirs, n_reps, n_bins)
    print 'min: %5.2f, max: %5.2f' % (rates.min(), rates.max())
    print '----------------'

# def scale_axes(axes, ymin=0, maxticks=4):
#     # get maximum scale value
#     assert type(axes) == list
#     ulim = None
#     llim = 1e9
#     for ax in axes:
#         llim = min(llim, ax.get_ylim()[0])
#         ulim = max(ulim, ax.get_ylim()[1])
#     #ulim *= 1.05
#     print 'scaling axes from:', llim, 'to', ulim
#     for ax in axes:
#         ax.set_xlim(xmin=0)
#         if ymin == 'auto':
#             ymin=llim
#         ax.set_ylim(ymin=ymin, ymax=ulim)

#         #yt = ax.get_yticks()
#         #nyt = [yt[0], 0, yt[-1]]
#         #ax.set_yticks(nyt)
#         #ax.set_yticks(yt)
#         #ax.set_xticks([0,1.])
#         #ax.xaxis.set_ticklabels([])
#         #ax.set_yticks([0,1.])
#         #ax.yaxis.set_ticklabels([])
#         #if len(current_yticks) > 5:
#         #ax.set_yticks([])



