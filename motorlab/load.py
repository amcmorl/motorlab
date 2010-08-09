from git import Repo
from warnings import warn
from glob import glob
from socket import gethostname
from motorlab.tuning_change.efile import list_units

days_in_set = ['081309','081409','081709','081809','081909']
data_locations = \
    { 'amcopti'  : "/home/amcmorl/files/pitt/frank/data/",
      'amcnote2' : "/home/amcmorl/files/pitt/frank/offline_sorted_multiday/" }

def check_git_status(branch):
    hostname = gethostname()
    if hostname == 'amcopti':
        # only amcopti uses a git-based data source
        repo = Repo(data_locations[hostname])
        if not repo.active_branch == branch:
            raise EnvironmentError("Active branch in data git repository is "
                   "%s, not %s as required." % (repo.active_branch, branch))

def get_file_list(days=days_in_set, data_locs=data_locations,
                  git_branch='offline_sorted_multiday'):
    if git_branch == 'offline_sorted_multiday':
        warn('default repository will be switched to master')
    data_dir = data_locs[gethostname()]
    file_pat = 'Frank.VR.*.CenterOut.mat'
    if git_branch != None:
        check_git_status(git_branch)
    files = []
    for day in days:
        files.extend(glob(data_dir + day + '/' + file_pat))
    files.sort()
    return files

def get_units_list(days=days_in_set, data_locs=data_locations,
                   git_branch='master'):
    file0 = get_file_list(days=days, data_locs=data_locs,
                          git_branch=git_branch)[0]
    return list_units(file0)

def pick_units(units, cond='>100'):
    '''
    Select a subset of units based on `condition` relative to their unit_num.
    '''
    out = []
    for unit in units:
        unit_num, chan_num = unit.split('_')
        unit_num = int(unit_num[4:]) # assumes has 'Unit' on the front
        if eval(str(unit_num) + cond):
            out.append(unit)
    return out

