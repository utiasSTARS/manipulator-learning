import numpy as np

from manipulator_learning.sim.envs.rewards.reach import dist_tanh


TABLE_HEIGHT = .6247


def lift_dense(obj_pos, ee_pos, height_for_suc, lift_multiplier=3, reach_multiplier=1, tanh_reach_multiplier=5,
                bottom_height=TABLE_HEIGHT):
    des_height = height_for_suc + .03  # add 3cm to height needed for success condition
    obj_bottom_dist = obj_pos[2] - bottom_height
    obj_bottom_dist_scaled = obj_bottom_dist / des_height
    reward = lift_multiplier * min(obj_bottom_dist_scaled, 1.0) + \
             reach_multiplier * dist_tanh(obj_pos, ee_pos, tanh_reach_multiplier)
    return reward


def lift_dense_multiple(obj_poss, ee_pos, height_for_suc, lift_multiplier=3, reach_multiplier=1,
                        tanh_reach_multiplier=5, bottom_height=TABLE_HEIGHT):
    reward = 0
    for o_pos in np.atleast_2d(obj_poss):
        reward += lift_dense(o_pos, ee_pos, height_for_suc, lift_multiplier, reach_multiplier, tanh_reach_multiplier,
                             bottom_height)
    return reward


def lift_sparse(obj_pos, height_for_suc, bottom_height=TABLE_HEIGHT):
    obj_bottom_dist = obj_pos[2] - bottom_height
    return obj_bottom_dist >= height_for_suc


def lift_sparse_multiple(obj_poss, height_for_suc, bottom_height=TABLE_HEIGHT):
    all_suc = True
    for o_pos in np.atleast_2d(obj_poss):
        if not lift_sparse(o_pos, height_for_suc, bottom_height):
            all_suc = False
            break
    return all_suc
