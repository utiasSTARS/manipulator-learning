import numpy as np


def hold_timer(suc_bool, ep_timesteps, real_t_per_ts, min_time, start_time):
    """ Get result of a timer to decide whether a success reward criteria has been met for long enough """
    if ep_timesteps <= 1:
        start_time = None  # acts as a reset

    done_success = False
    if suc_bool:
        if start_time is None:
            start_time = ep_timesteps
        elif (ep_timesteps - start_time) * real_t_per_ts > min_time:
            done_success = True
    else:
        start_time = None

    return done_success, start_time


def get_world_ee_pose(env):
    return env.gripper.manipulator.get_link_pose(env.gripper.manipulator._tool_link_ind)


def get_world_obj_pose(env, obj_id):
    pos, quat = env._pb_client.getBasePositionAndOrientation(obj_id)
    return (*pos, *quat)


def get_done_suc_fail(dist, reward, limit_reached, dense_reward, env_obj):
    done_success, done_failure = False, False
    if dist < env_obj.reach_radius:
        if env_obj.reach_radius_start_time is None:
            env_obj.reach_radius_start_time = env_obj.ep_timesteps
        elif (env_obj.ep_timesteps - env_obj.reach_radius_start_time) * env_obj.real_t_per_ts > env_obj.reach_radius_time:
            done_success = True
    else:
        env_obj.reach_radius_start_time = None

    if env_obj.limits_cause_failure and limit_reached:
        done_success = False
        done_failure = True

    if env_obj.success_causes_done and done_success:
        reward = env_obj.done_success_reward
    if env_obj.failure_causes_done and done_failure:
        reward = env_obj.done_failure_reward

    if dense_reward:
        return reward, done_success, done_failure
    else:
        return done_success, done_success, done_failure