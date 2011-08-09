import numpy as np
import vectors

'''
Utility functions for handling formatted data (from the RP3 experiment).
'''
# codes assigned by executive for each task phase
TE_HAND_OPENING = 20
TE_REACHING = 6
TE_HOMING = 7

get_data = lambda d,v,t: np.asarray(d[v], dtype=t).squeeze()

def get_pts(fData, trials=None, phases=None):
    '''
    get boolean vector index into time-series data corresponding to
    the given trials and phases
    '''
    trial_no = get_data(fData, 'TrialNo', int)
    if trials != None:
        trial_pts = np.in1d(trial_no, trials)
    else:
        # all true by default
        trial_pts = np.ones_like(trial_no, dtype=bool)
    task_state_codes = get_data(fData, 'TaskStateCodes/Values', int)
    if phases != None:
        phase_pts = np.in1d(task_state_codes, phases)
    else:
        # all true by default
        phase_pts = np.ones_like(task_state_codes, dtype=bool)
    return trial_pts & phase_pts
    
def get_translation_dist_to_targ(fData, pt=None, planner=False):
    '''
    '''
    if pt == None:
        pt = slice(None)

    if planner:
        target_field = 'Position/PlannerTarget'
    else:
        target_field = 'Position/target'
    targ_pos = np.asarray(fData[target_field])[pt,0:3]
    curr_pos = np.asarray(fData['Position/Actual'])[pt,0:3]
    dist = vectors.norm(targ_pos - curr_pos, axis=1)
    return dist
