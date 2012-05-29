import numpy as np
import motorlab.tuning.calculate.kinematics as kin
from vectors import unitvec

from numpy.testing import assert_array_equal

def test_get_vel():
    pos = np.array([[ 0, 0,  0],
                    [ 1, 2, -1],
                    [ 2, 4, -2],
                    [ 3, 6, -3]]) # shape (4,3)
    time = np.arange(pos.shape[0])

    # default case
    vel = kin.get_vel(pos, time, tax=0, spax=1)
    correct_vel = np.array([[ 1, 2, -1],
                            [ 1, 2, -1],
                            [ 1, 2, -1]])
    assert_array_equal(vel, correct_vel)

    # case where space axis is first
    vel = kin.get_vel(pos.T, time[None], tax=1, spax=0)
    assert_array_equal(vel, correct_vel.T)

    # case with additional axes
    pos = np.tile(pos[None], (2,1,1))
    time = np.tile(time[None], (2,1))
    vel = kin.get_vel(pos, time, tax=1, spax=2)
    correct_vel = np.tile(correct_vel[None], (2,1,1))
    assert_array_equal(vel, correct_vel)

def test_get_dir():
    pos = np.array([[ 0, 0,  0],
                    [ 1, 2, -1],
                    [ 2, 4, -2],
                    [ 3, 6, -3]]) # shape (4,3)
    dir = kin.get_dir(pos, tax=0, spax=-1)
    correct_dir = unitvec(np.array([[1,2,-1], [1,2,-1], [1,2,-1]]), axis=1)

    # default case
    assert_array_equal(dir, correct_dir)

    # switched axes case
    dir = kin.get_dir(pos.T, tax=1, spax=0)
    assert_array_equal(dir, correct_dir.T)

    # additional axes case
    pos = np.tile(pos[None], (2,1,1))
    dir = kin.get_dir(pos, tax=1, spax=-1)
    correct_dir = np.tile(correct_dir[None], (2,1,1))
    assert_array_equal(dir, correct_dir)

def test_get_speed():
    pos = np.array([[ 0., 0,  0],
                    [ 1, 2, -1],
                    [ 2, 4, -2],
                    [ 3, 6, -3]]) # shape (4,3)
    time = np.arange(pos.shape[0])
    speed = kin.get_speed(pos, time, tax=0, spax=-1)
    rt6 = np.sqrt(6)
    correct_speed = np.array([rt6, rt6, rt6])

    # default case
    assert_array_equal(speed, correct_speed)

    # switched axes case
    speed = kin.get_speed(pos.T, time[None], tax=1, spax=0)
    correct_speed = np.array([rt6, rt6, rt6])
    assert_array_equal(speed, correct_speed)

    # additional axes case
    pos = np.tile(pos[None], (2,1,1))
    time = np.tile(time[None], (2,1))
    speed = kin.get_speed(pos, time, tax=1, spax=2)
    correct_speed = np.tile(correct_speed, (2,1))
    assert_array_equal(speed, correct_speed)
