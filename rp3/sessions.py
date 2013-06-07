from glob import glob
import os
import motorlab.rp3.xm_file as xm
import numpy as np

verbose = False

def get_data_dir():
    return os.environ['ROBOTDATA']

def get_subject_sessions(subject, data_dir=None, threshold=70,
                          first_session=0, last_session=np.inf):
    '''
    Get all the valid sessions performed by `subject`.
    Valid is defined as containing more than `threshold` trials.

    Parameters
    ----------
    subject : string
      name of subject
    data_dir : string, optional
      path to directory containing subject directories, defaults to
      os.environ['ROBOTDATA']
    threshold : int, optional
      number of data files that need to be present to qualify
    first_session, last_session : int, optional
      number of first and last sessions to include
    '''
    if type(subject) != str:
        raise ValueError('`subject` must be of type `str`')
    if data_dir == None:
        data_dir = get_data_dir()
    elif type(data_dir) != str:
        raise ValueError('`data_dir` must be of type `str`, or None')


    subject_path = data_dir + '/' + subject
    lookfor = '%s/Raw/%s.DK.?????' % (subject_path, subject)
    sessions = glob(lookfor)
    if len(sessions) == 0:
        raise ValueError('No sessions found looking for %s' % (lookfor))

    valid_sessions = []
    for session in sessions:
        session_base = os.path.basename(session)
        XM_files = xm.get_XM_files(session_base, data_dir)
        session_num = int(session.split('.')[-1])
        criteria = (XM_files >= threshold) \
            & (session_num >= first_session) \
            & (session_num <= last_session)
        if criteria:
            valid_sessions.append(session_base)
    valid_sessions.sort()
    return valid_sessions
