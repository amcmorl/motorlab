import numpy as np

def OLE(B, sigma, f, b0, m, method='variance_only'):
    '''
    Optimal Linear Estimator reconstruction algorithm
    
    Parameters
    ----------
    B : ndarray
      matrix of nx3 regression coefficients from multiple linear regression
    sigma : ndarray
      n x n covariance matrix of residuals from regression to get B
    f : ndarray
      vector of n firing rates
    b0 : ndarray
      vector of n baseline firing rates
      
    Notes
    -----
    OLE is defined in ﻿Chase SM, Schwartz AB & Kass RS (2009). Bias, 
    optimal linear estimation, and the differences between open-loop 
    simulation and closed-loop performance of spiking-based 
    brain-computer interface algorithms. Neural Networks 22, 1203–1213.
    
    .. math::
    
        \vec{r} = B\vec{d} = \frac{f_t - b}{m}
        \Rightarrow Bm\vec{d} = f_t - b_0
        \Rightarrow Bm\vec{d} + b_0 = f_t
        \Rightarrow m\frac{\beta}{m}\vec{d} + b_0 = f_t
        
    '''
    if method == 'minimal':
        # double diag correctly generates 2-d matrix
        sigma = 1
    elif method == 'full':
        pass # use sigma as provided
    elif method == 'variance-only':
        sigma = np.diag(np.diag(sigma))
    else:
        raise ValueError('method %s not known' % method)
        
    B = np.asmatrix(B)
    pinv = np.linalg.pinv
    PD = alpha * pinv(B.T * pinv(sigma) * B) * B.T * pinv(sigma)
    r = (f - b0) / m
    d = PD * r
    return d

def uniform_rvs_cart(size=1, ndim=3):
    '''
    Notes
    -----
    Tested visually for 3d and using histogram for 2d
    '''
    xyz = np.random.normal(size=(size, ndim))
    return xyz / np.sqrt(np.sum(xyz**2, axis=1))[:,None]

def test_OLE():
    nunit = 100
    # make a dummy population of perfectly tuned neurons
    pds = uniform_rvs_cart(size=nunit)
