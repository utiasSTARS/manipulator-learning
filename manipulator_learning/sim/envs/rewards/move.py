import numpy as np


def move_dense(obj_vel, obj_acc, ee_pos, reach_multiplier=1, tanh_reach_multiplier=5):
    # todo implement if needed
    return 0


def move_sparse(obj_t_vel, obj_t_acc, vel_min=.05, acc_max=5.0):
    obj_t_vel_mag = np.linalg.norm(obj_t_vel)
    obj_t_acc_mag = np.linalg.norm(obj_t_acc)

    return obj_t_vel_mag > vel_min and obj_t_acc_mag < acc_max
