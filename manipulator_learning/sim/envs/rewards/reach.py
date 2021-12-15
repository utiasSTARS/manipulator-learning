import numpy as np


def dist_tanh(obj0_pos, obj1_pos, multiplier=5.0):
    dist = np.linalg.norm(np.array(obj0_pos) - np.array(obj1_pos))
    return 1 - np.tanh(multiplier * dist)


def reach_sparse(obj0_pos, obj1_pos, dist_threshold):
    dist = np.linalg.norm(obj0_pos - obj1_pos)
    return dist < dist_threshold