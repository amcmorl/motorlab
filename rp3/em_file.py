import os.path
from glob import glob

def get_file_number(em_file):
    return int(os.path.basename(em_file).split('.')[2])

def get_latest_em_file(direc, min_num=0):
    '''
    Parameters
    ----------
    direc : string
      directory to search in
    '''
    file_pat = 'EM.default'
    full_file_pat = '%s/%s.*.mat' % (direc, file_pat)
    em_files = glob(full_file_pat)
    em_files.sort()
    if len(em_files) > 0:
        latest = em_files[-1]
        if get_file_number(latest) >= min_num:
            return latest
    return None
