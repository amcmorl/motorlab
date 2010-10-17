from git import Repo
from warnings import warn
from glob import glob
from socket import gethostname
from motorlab.data_files import list_units

days_in_set = ['081309','081409','081709','081809','081909']
data_locations = \
    { 'amcopti'  : "/home/amcmorl/files/pitt/frank/data/",
      'amcnote2' : "/home/amcmorl/files/pitt/frank/offline_sorted_multiday/" }
uses_git = ['amcopti']

def check_git_status(branch='any'):
    hostname = gethostname()
    if hostname in uses_git:
        repo = Repo(data_locations[hostname])
        if branch == 'any':
            warn("Active branch in data repo is %s" % (repo.active_branch))
        elif not repo.active_branch == branch:
            raise EnvironmentError("Active branch in data repo is"
                   "%s; %s required." % (repo.active_branch, branch))

def get_file_list(days=days_in_set, data_locs=data_locations,
                  git_branch='any'):
    check_git_status(git_branch)
    data_dir = data_locs[gethostname()]
    file_pat = 'Frank.VR.*.CenterOut.mat'
    files = []
    for day in days:
        files.extend(glob(data_dir + day + '/' + file_pat))
    files.sort()
    return files

def get_units_list(days=days_in_set, data_locs=data_locations,
                   git_branch='any'):
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
