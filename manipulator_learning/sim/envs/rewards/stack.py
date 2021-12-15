import numpy as np

from manipulator_learning.sim.envs.rewards.reach import dist_tanh


def stack_dense(stack_contact_bool, obj_poss, obj_height, ee_pos, stack_mult=10, stack_pos_mult=3, reach_mult=1,
                tanh_reach_multiplier=5):
    """ Currently only set up for stacking of one block onto another. Assumes stacking obj 0 on obj 1. """
    reach_rew = dist_tanh(obj_poss[0], ee_pos, tanh_reach_multiplier)
    b2b_rew = dist_tanh(obj_poss[0] + np.array([0, 0, obj_height]), obj_poss[1], tanh_reach_multiplier)
    return stack_mult * stack_contact_bool + stack_pos_mult * b2b_rew + reach_mult * reach_rew


def stack_sparse(pb_client, obj_pb_ids, arm_pb_id, table_pb_id):
    """ Need to satisfy 3 conditions to be a stack: block to block contact, no block to table contact, and no
    block to end-effector contact

    Assuming stacking obj 0 onto 1, 1 onto 2, etc."""
    # only need block to ee for top object
    o2ee_contact = len(pb_client.getContactPoints(obj_pb_ids[0], arm_pb_id)) > 0

    o2o_contacts = []
    o2t_contacts = []
    # for i in range(len(obj_pb_ids) - 1, 0, -1):
    for i in range(0, len(obj_pb_ids) - 1):
        o2o_contacts.append(len(pb_client.getContactPoints(obj_pb_ids[i], obj_pb_ids[i + 1])) > 0)
        # bottom block, aka highest index, can touch table
        o2t_contacts.append(len(pb_client.getContactPoints(obj_pb_ids[i], table_pb_id)) > 0)
    all_o2o_contact = len(o2o_contacts) > 0 and all(o2o_contacts)
    no_o2t_contact = not any(o2t_contacts)

    return all_o2o_contact and no_o2t_contact and not o2ee_contact
