import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#==============================================================================
# mplot3d
#==============================================================================

def plot_value_mplot3d(val, ax=None, **kwargs):
    '''
    Parameters
    ----------
    val : ndarray
      shape (ntrial, nbin, 3)
    '''
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for trial in val:
        t = trial.T
        plt.plot(t[0], t[1], t[2], **kwargs)
    
#==============================================================================
# mplot2d
#==============================================================================
        
def plot_value_2d(val, ax=None, **kwargs):
    '''
    Parameters
    ----------
    val : ndarray
      shape (ntrial, nbin, 2)
    '''
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    for trial in val:
        t = trial.T
        plt.plot(t[0], t[1], **kwargs)
    
