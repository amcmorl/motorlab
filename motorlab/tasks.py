import numpy as np

def make_26_targets():
    ''' Generate target positions for 26-target center-out task.

    Returns
    -------
    pts : ndarray
      pts on square, pre-normalizing to unit radius (task length)
      shape (26, 3)
    spts : ndarray
      x,y,z points on a sphere of radius sqrt(3.)
      shape (26, 3)
    circs : ndarray
      indices into `pts` or `spts` of points on each polar (i.e. N & S)
      great circle, shape (4, 8)
    '''
    # corners
    pts = np.mgrid[-1:2:1,-1:2:1, -1:2:1].reshape(3,27).T
    radii = np.sqrt(np.sum(pts**2, axis=1))
    pts = pts[radii > 0,...]
    radii = radii[radii > 0]

    # make on radius sqrt(3) sphere, and get rid of origin
    spts = pts / radii[...,None] * np.sqrt(3)

    circs = np.array([[13, 16, 15, 14, 12, 9, 10, 11],
                      [13, 25, 24, 23, 12, 0,  1,  2],
                      [13, 22, 21, 20, 12, 3,  4,  5],
                      [13, 19, 18, 17, 12, 6,  7,  8]])
    
    return pts, spts, circs
