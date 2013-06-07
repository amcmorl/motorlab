import h5py
import os
from glob import glob
from scipy import io

verbose = False

def descend_fields(d, field):
    # assume that XM was specified in `field`, now skip it
    field_list = field.split('/')[1:]
    # descend field tree
    for fname in field_list:
        f = d.__dict__[fname]
    return f

def get_XM_fields(path, fields):
    '''
    Retrieve `field` from an XM file, either old or new (hdf5) format.

    Parameters
    ----------
    path : string
      path to XM file to load
    fields : string or list of string
      names of field to load, in the format 'XM/config/task_state_config/...'
    '''
    #print path, fields, '\n'

    if not h5py.is_hdf5(path):
        # assume XM is first field
        d = io.loadmat(path, struct_as_record=False, squeeze_me=True)['XM']

        if type(fields) == list:
            fs = []
            for field in fields:
                try:
                    fs.append(np.asarray(_descend_fields(d, field)))
                except KeyError:
                    fs.append(None)
            return fs
        else: # want only one field
            try:
                ret = np.asarray(_descend_fields(d, fields))
                return ret
            except KeyError:
                return None

    else: # is an hdf5 file
        d = h5py.File(path)
        if type(fields) == list:
            fs = []
            for field in fields:
                fs.append(np.asarray(d[field]))
            return fs
        else: # want only one field
            return np.asarray(d[fields])

def get_XM_files(session, data_dir, min_files=60):
    '''
    Parameters
    ----------
    session : string
      name of session to process
      we'll assume that the characters before the first '.' are the subject directory
    data_dir : string, optional
      base directory to search for session directory
    '''
    if type(session) != str:
        raise ValueError('`session` must be of type `str`')
    if type(data_dir) != str:
        raise ValueError('`data_dir` must be of type `str`, or None')

    subject = session.split('.')[0]
    path = '%s/%s/Raw/%s' % (data_dir, subject, session)
    if not os.path.isdir(path):
        raise ValueError('%s is not a valid session directory' % (path))

    # oldest, v1 naming convention
    lookfor = '%s/%s.XM.Trial*???.mat' % (path, subject)
    XM_files = glob(lookfor)
    if len(XM_files) > min_files:
        file_version = 1
        XM_files.sort()
        return XM_files, file_version

    # v2 naming convention
    lookfor = '%s/XM.Trial*???.mat' % (path)
    XM_files = glob(lookfor)
    if len(XM_files) > min_files:
        XM_files.sort()
        file_version = 2
        return XM_files, file_version

    # newest, v3 naming convention
    lookfor = '%s/XM.*???.mat' % (path)
    XM_files = glob(lookfor)
    if len(XM_files) > min_files:
        file_version = 3
        XM_files.sort()
        return XM_files, file_version

    if len(XM_files) == 0:
        if verbose:
            warn('No files found looking for %s' % (lookfor))

    return [], 0
