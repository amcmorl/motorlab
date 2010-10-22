from git import Repo
from warnings import warn
from glob import glob
from socket import gethostname
from motorlab.data_files import list_units

'''
Philosophy is that there is one data store per machine,
located at `locations[hostname]`, which may or may not be under
version control.

Then there are multiple definitions of datasets,
which are collections of files or days.
'''
# locations
locations = {'amcopti'  : "/home/amcmorl/files/pitt/frank/data/",
             'amcnote2' : \
                 "/home/amcmorl/files/pitt/frank/offline_sorted_multiday/"}
uses_git = ['amcopti']

# datasets
# osmd = offline sorted multiday
osmd = {'days' : ['081309','081409','081709','081809','081909'],
        'git' : 'offline_sorted_multiday',
        'aliases' : ['offline sorted multiday', 'osmd',
                     'offline_sorted_multiday']}

datasets = {'offline sorted multiday' : osmd}
# generate dataset_aliases
aliases = {}
for name, dset in datasets.iteritems():
    if 'aliases' in dset.keys():
        for alias in dset['aliases']:
            aliases[alias] = name
        
def check_git_status(branch='any'):
    '''
    Check git branch of `location` on hostname.
    '''
    hostname = gethostname()
    if hostname in uses_git:
        repo = Repo(locations[hostname])
        if branch == 'any':
            warn("Active branch in data repo is %s" % (repo.active_branch))
        elif not repo.active_branch == branch:
            raise EnvironmentError("Active branch in data repo is"
                   "%s; %s required." % (repo.active_branch, branch))

def get_file_list(dataset='offline sorted multiday'):
    '''
    '''
    dset_name = aliases[dataset]
    dset = datasets[dset_name]
    check_git_status(branch=dset['git'])
    data_dir = locations[gethostname()]
    file_pat = 'Frank.VR.*.CenterOut.mat'
    files = []
    if 'files' in dset.keys():
        files.extend(dset['files'])
    if 'days' in dset.keys():
        for day in dset['days']:
            files.extend(glob(data_dir + day + '/' + file_pat))
    files.sort()
    return files

def get_units_list(dataset='offline sorted multiday'):
    file0 = get_file_list(dataset=dataset)[0]
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
