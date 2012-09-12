import numpy as np
from motorlab.tuning.fit import glm_any_model, unpack_coefficients

def get_sort_by_task_dir(tasks):
    '''Sort tasks by direction, in order of angle from x,y, and lastly z.

    Parameters
    ----------
    tasks : array_like
      shape (n_tasks, 6)
    '''
    starts, stops = tasks[:,:3], tasks[:,3:]
    dirs = stops - starts
    xyz = np.array(([1,0,0], [0,1,0],[0,0,1]))
    angs = np.dot(dirs,xyz)
    return np.lexsort(angs.T)

def get_sort_by_pd(bnd):
    '''
    Parameters
    ----------
    bnd : cell.BinnedData instance
      contains PSTHs (n, ...) and positions (n, ...) from construct_PSTHs
    tasks : array_like
      shape (n, 6), start and target positions of movements

    Returns
    -------
    indices to sort PSTH tasks relative to PD.
    '''
    #    bnd.ensure_flat_inplace()
    tasks = bnd.tasks
    starts, stops = tasks[:,:3], tasks[:,3:]
    dirs = stops - starts
    ndset = bnd.count.shape[2]    
    pds = np.zeros((ndset, 3))    

    for i in xrange(ndset):
        count, pos, time = bnd.get_for_glm(i)
        model = 'kd'
        b, cov = glm_any_model(count, pos, time, model=model)
        bdict = unpack_coefficients(b, model=model)
        pds[i] = bdict['d'] / np.linalg.norm(bdict['d'])
        
    angs = np.arccos(np.dot(pds, dirs.T))
    return np.argsort(angs, axis=-1)

def get_sort_by_pd_co(bnd):
    '''
    Parameters
    ----------
    bnd : BinnedData
    
    Returns
    -------
    sort : array
      indices by which to sort bnd to sort by preferred direction and
      center-out/out-center, shape = (nunit, ntask)
    '''
    tasks = bnd.tasks
    pd_ord = get_sort_by_pd(bnd)
    co = get_center_out(tasks) # center-out trials, in `tasks` order
    copd = co[pd_ord]          # center-out trials, in pd order
                               # means ~copd = out-center trials, in pd order
    nunit = bnd.PSTHs.shape[2]
    copd_ord = pd_ord[copd].reshape(nunit, -1) # center-out trials, in pd order
    ocpd_ord = pd_ord[~copd].reshape(nunit, -1) # out-center trials, in pd order
    return np.concatenate((copd_ord, ocpd_ord), axis=1)

def get_center_out(tasks):
    '''
    Return boolean index array giving center-out trials.

    Parameters
    ----------
    tasks : ndarray
      start and finish co-ordinates of task, shape (ntask, ndim * 2)

    Returns
    -------
    co : ndarray
      boolean array, True for center-out tasks, shape (ntask)
      
    Notes
    -----
    Use `~get_co` to get out-center trials.
    '''
    return np.all(tasks[:,0:3] == 0, axis=1)

def get_sort_zsubset(tasks):
    '''
    Sort tasks into grids by z direction.

    Returns
    -------
    out_front, out_mid, out_back : ndarray
      indices into `tasks` for center-out trials
    in_back, in_mid, in_front : ndarray
      indices into `tasks` for out-center trials
    '''
    # center-out first
    co = get_center_out(tasks)
    out_idx = np.where(co)[0]
    out_zord = np.argsort(tasks[co, -1]) # sort by z

    # out-center is not center-out
    in_idx = np.where(~co)[0]
    in_zord = np.argsort(tasks[~co, 2]) # sort by z
    
    # define grids by index into zord
    front, mid, back = range(1,9), range(9,17), range(17,25)

    # flip front and back for oc (vs co) to sort b/c sorting by direction
    return out_idx[out_zord[front]], out_idx[out_zord[mid]], \
        out_idx[out_zord[back]], \
        in_idx[in_zord[back]], in_idx[in_zord[mid]], \
        in_idx[in_zord[front]]

def sort_by_pd_co(bnd):
    cobypd = get_sort_by_pd_co(bnd)
    return bnd.PSTHs[cobypd]

