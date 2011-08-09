import matplotlib.pyplot as plt
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D

def plot_position_mplot3d(pos, ax=None, **kwargs):
    '''
    Parameters
    ----------
    pos : ndarray
      shape (ntrial, nbin, 3)
    '''
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for trial in pos:
        t = trial.T
        plt.plot(t[0], t[1], t[2], **kwargs)
