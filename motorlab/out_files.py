import os
import time

home = os.environ['HOME']
pitt = '/files/pitt'

project_dirs = {'networks' : home + pitt + '/networks',
                'tuning_change' : home + pitt + '/tuning_change'}

def make_today_dir(project='tuning_change'):
    outputs = project_dirs[project] + '/run'
    today = outputs + '/' + time.strftime('%y%m%d')
    if not os.path.exists(today):
        os.mkdir(today)
    return today

def now_str():
    return time.strftime('%H%M%S')

def get_out_name(save_dir, what):
    return save_dir + '/' + what + '_' + now_str()
