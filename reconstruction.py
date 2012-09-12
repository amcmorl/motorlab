import numpy as np

def reconstruct_direction(f, b, m):
    pass

def OLE(B, sigma, f, m, b0, method='variance-only'):
    '''
    Optimal Linear Estimator reconstruction algorithm
    
    Parameters
    ----------
    B : ndarray
      matrix of nunit x ndim regression coefficients from OLS
    sigma : ndarray or None
      nunit x nunit covariance matrix of residuals from OLS, to get B
    f : ndarray
      vector of nunit firing rates
      OR
      matrix of nsamp x nunit firing rates
    m : ndarray
      vector of nunit modulation depths, given by :math:`|\beta|`
    b0 : ndarray
      vector of nunit baseline firing rates

    Returns
    -------
    d : ndarray

    Notes
    -----
    OLE is defined in Chase SM, Schwartz AB & Kass RS (2009). Bias, 
    optimal linear estimation, and the differences between open-loop 
    simulation and closed-loop performance of spiking-based 
    brain-computer interface algorithms. Neural Networks 22, 1203-1213.
    
    .. math::
    
        \vec{r} = B\vec{d} = \frac{f_t - b}{m}
        \Rightarrow Bm\vec{d} = f_t - b_0
        \Rightarrow Bm\vec{d} + b_0 = f_t
        \Rightarrow m\frac{\beta}{m}\vec{d} + b_0 = f_t
        
    '''
    methods = ['minimal', 'full', 'variance-only']
    if not method in methods:
        raise ValueError('method %s not known. Available methods ' \
                         'are %s' % (method, ', '.join(methods)))
    n = B.shape[0]
    if method == 'minimal':
        # double diag correctly generates 2-d matrix
        sigma = np.identity(n, dtype=float)
    elif method == 'full':
        pass # use sigma as provided
        if type(sigma) != np.ndarray:
            raise ValueError('for method "full", sigma must be of type ndarray')
    elif method == 'variance-only':
        if type(sigma) != np.ndarray:
            raise ValueError('for method "variance-only", sigma must be of type ndarray')
        sigma = np.diag(np.diag(sigma))
    else:
        raise ValueError('this should never happen')
        
    # matrix allows easier writing of computations
    B = np.asmatrix(B)
    pinv = np.linalg.pinv
    PD = pinv(B.T * pinv(sigma) * B) * B.T * pinv(sigma)
    #alpha = ...
    r = (f - b0[:,None]) / m[:,None]
    d = PD * r
    return np.asarray(d)

def test_OLE():
    def angle2cart(ang, axis=-1):
        return np.apply_along_axis(_angle2cart, axis, ang)

    def _angle2cart(ang):
        return np.cos(ang), np.sin(ang)

    enspve = lambda x : np.clip(x, 0, np.inf)
    
    ndir = 8
    #nunit = 20
    #ndim = 2
    #std_m = 10
    #mean_m = 0
    #mean_b0 = 40
    #beta = np.random.normal(size=(nunit, ndim)) * std_m + mean_m
    #b0 = np.random.normal(size=nunit) * mean_b0 / 3. + mean_b0
    
    # some test data - originally randomly generated
    beta = np.array([[ -4.1167318 ,  5.70386829],
                     [ 11.07085419,  6.25681314],
                     [ -4.50656523,  1.82404699],
                     [ -5.3357364 , -1.67050185],
                     [-13.7570496 ,  9.93104535],
                     [ 14.24775439,  0.42718023],
                     [  9.50523668, -3.44359843],
                     [ -4.46437841,  2.6127224 ],
                     [ -4.39217573, -4.98102944],
                     [ -9.12123191,  5.94525391],
                     [  5.98203818,  6.2369812 ],
                     [ -7.62861248, -4.5344569 ],
                     [  7.94526162, -7.64013987],
                     [ 10.93186065, -0.69518603],
                     [ 13.60093124, -4.70488677],
                     [ -7.90036757,-13.27519872],
                     [  5.83727905,  7.14287259],
                     [  3.31767414,-11.03132855],
                     [  2.15196874, 10.52988149],
                     [ 16.55251712,-19.76089961],])
    b0 = np.array([41.20992642, 29.69643519, 33.2307665,  49.1259394,
                   27.60366162, 62.91077507, 26.78056616, 27.70454193,
                   41.59267956, 34.24925522, 51.2161271,  10.99736431,
                   29.34859964, 32.84940301, 28.80518506, 37.013994,
                   16.12532599, 36.94383252, 37.80836605, 58.4344842])
    nunit = beta.shape[0]
    
    m = np.sqrt(np.sum(beta**2, axis=1))
    B = beta / m[:,None]
    angle = np.linspace(0, np.pi * 2, ndir, endpoint=False)
    targ = angle2cart(angle[:,None])
    fr = enspve(np.dot(beta, targ.T) + b0[:,None])
    print "fr shape:", fr.shape
    
    # test is to recover target directions from firing rates and 
    # regression results
    
    # perfect data => minimal should work
    d = OLE(B, None, fr, m, b0, method='minimal')
    assert np.allclose(targ, d.T)
    
    # perfect data => sigma = identity should also work
    sigma = np.identity(nunit)
    d = OLE(B, sigma, fr, m, b0, method='variance-only')
    print "d  shape:", d.shape
    assert np.allclose(targ, d.T)

