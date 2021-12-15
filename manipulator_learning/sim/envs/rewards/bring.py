import numpy as np

from manipulator_learning.sim.envs.rewards.reach import dist_tanh


def bring_dense(obj_pos, ee_pos, target_pos, bring_multiplier=3, reach_multiplier=1, tanh_reach_multiplier=5,
                insert_bonus=False, insert_multiplier=10):
    reach_rew = dist_tanh(obj_pos, ee_pos, tanh_reach_multiplier)
    bring_rew = dist_tanh(obj_pos, target_pos, tanh_reach_multiplier)
    insert_rew = int(insert_bonus)
    return insert_multiplier * insert_rew + bring_multiplier * bring_rew + reach_multiplier * reach_rew


def bring_dense_multiple(obj_poss, ee_pos, target_poss, bring_multiplier=3, reach_multiplier=1,
                        tanh_reach_multiplier=5, insert_bonuss=(), insert_multiplier=10):
    reward = 0
    for i, (o_pos, t_pos) in enumerate(zip(np.atleast_2d(obj_poss), np.atleast_2d(target_poss))):
        if len(insert_bonuss) > 0:
            in_bonus = insert_bonuss[i]
        else:
            in_bonus = False
        reward += bring_dense(o_pos, ee_pos, t_pos, bring_multiplier, reach_multiplier, tanh_reach_multiplier,
                              in_bonus, insert_multiplier)
    return reward


def bring_sparse(obj_pos, target_pos, dist_threshold):
    dist = np.linalg.norm(obj_pos - target_pos)
    return dist < dist_threshold


def bring_sparse_multiple(obj_poss, target_poss, dist_threshold):
    all_suc = True
    for o_pos, t_pos in zip(obj_poss, target_poss):
        if not bring_sparse(o_pos, t_pos, dist_threshold):
            all_suc = False
            break
    return all_suc


def bring_sparse_multiple_list(obj_poss, target_poss, dist_threshold, contact_list=None):
    sucs = []
    for o_i, (o_pos, t_pos) in enumerate(zip(np.atleast_2d(obj_poss), np.atleast_2d(target_poss))):
        sucs.append(bring_sparse(o_pos, t_pos, dist_threshold))
        if contact_list is not None:
            sucs[-1] = sucs[-1] and contact_list[o_i]
    return sucs


def bring_contact_bonus(pb_client, obj_pb_ids, arm_pb_id, table_pb_id):
    """ For some bring goals, may be useful to also satisfy an object touching table and
    not touching arm condition. """
    o2ee_contacts = []
    o2t_contacts = []
    for o in obj_pb_ids:
        o2ee_contacts.append(len(pb_client.getContactPoints(o, arm_pb_id)) > 0)
        o2t_contacts.append(len(pb_client.getContactPoints(o, table_pb_id)) > 0)

    return not any(o2ee_contacts) and len(o2t_contacts) > 0 and all(o2t_contacts)


def bring_contact_bonus_list(pb_client, obj_pb_ids, arm_pb_id, table_pb_id):
    """ For some bring goals, may be useful to also satisfy an object touching table and
    not touching arm condition. """
    correct_contacts = []
    for o in obj_pb_ids:
        o2ee_contact = len(pb_client.getContactPoints(o, arm_pb_id)) > 0
        o2t_contact = len(pb_client.getContactPoints(o, table_pb_id)) > 0
        correct_contacts.append(not o2ee_contact and o2t_contact)

    return correct_contacts
