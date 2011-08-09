import numpy as np
import matplotlib.pyplot as plt

def plot_survival(survival):
    '''
    Plots the survival matrix across several days.
    
    Parameters
    ----------
    survival : ndarray
      object type array of unit survival
      shape ~ (N-1,I,J) where N is the number of days,
      a 1 in element i,j indicates that the neuron i in day n has survived as
        element j in day n+1
    '''
    fig = plt.figure()    
    ax = fig.add_subplot(111)    

    nsurv = survival.shape[0]    
    nday = nsurv + 1
    for d in xrange(nday):
        if d < nsurv:
            nunit = survival[d].shape[0]
        else:
            nunit = survival[d-1].shape[1]
        ax.plot(np.ones(nunit) * d, np.arange(nunit), 'ko', markersize=2)
    for d, s in enumerate(survival):
        I,J = s.nonzero()
        for i, j in zip(I,J):
            ax.plot(np.array([d, d+1]), np.array([i, j]), 'k-')
    return ax

def calculate_total_number_of_repeats(survival):
    '''
    Calculates the total number of repeats recorded for each identified unit.
    
    Parameters
    ----------
    survival : ndarray
      object type array of unit survival
      shape ~ (N-1,I,J) where N is the number of days,
      a 1 in element i,j indicates that the neuron i in day n has survived as
        element j in day n+1
    '''
    # going to have to give each unit a universal ID
    # (i.e. unique across all days)
    univ_id 