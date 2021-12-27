import argparse
from datetime import datetime

from manipulator_learning.sim.envs import *

from manipulator_learning.learning.imitation.device_utils import CollectDevice
from manipulator_learning.learning.imitation.collect_utils import *
import manipulator_learning.learning.data.img_depth_dataset as img_depth_dataset
from manipulator_learning.learning.utils.absorbing_state import Mask


parser = argparse.ArgumentParser()
parser.add_argument('--environment', type=str, default="ThingReachingXYState")
parser.add_argument('--directory', type=str, default='/tmp/demonstrations')
parser.add_argument('--demo_name', type=str, default=datetime.now().strftime("%y-%m-%d_%H-%M-%S"))
parser.add_argument('--device', type=str, default='gamepad')
parser.add_argument('--collect_interval', type=int, default=0)
parser.add_argument('--action_multiplier', type=float, default=.3)
parser.add_argument('--enforce_real_time', action='store_true', default=True,
                    help='if true, attempt to play environment at real time, based on .01s per'
                         'low level timestep and chosen n_substeps')
parser.add_argument('--ros_env_vr_teleop_help', action='store_true', default=False,
                    help='if true, when reset is called, user controls robot to assist in resetting the env'
                         'before the env is fully reset (e.g. to place objs in new positions).')
parser.add_argument('--show_opengl', action='store_true', help="show regular pybullet opengl renderer")
parser.add_argument('--save_on_success_only', action='store_true', help="always save on and only on success")

args = parser.parse_args()

ros_env = False
if 'ThingRos' in args.environment:
    from thing_gym_ros.envs import *
    ros_env = True
    env = globals()[args.environment](reset_teleop_available=args.ros_env_vr_teleop_help,
                                      success_feedback_available=True,
                                      high_ft_causes_failure=True,
                                      high_ft_causes_failure_thresh=[50, 15])
    env.render()

    # normal thing forward axis is y, up axis is z
    if 'Door' in args.environment:
        des_up_axis = (0, 0, -1)
    elif 'Drawer' in args.environment:
        des_up_axis = (1, 0, 0)
    else:
        des_up_axis = (0, 0, 1)

    dev = CollectDevice(args.device, valid_t_dof=env.valid_act_t_dof, valid_r_dof=env.valid_act_r_dof,
                        output_grip=env.grip_in_action, action_multiplier=args.action_multiplier,
                        des_up_axis=des_up_axis, des_forward_axis=(0, 1, 0))
    if args.device == 'vr':
        dev.dev.vel_pid.Kp = 0.7

    args.save_on_success_only = True
else:
    env = globals()[args.environment](action_multiplier=1.0, egl=not args.show_opengl, render_opengl_gui=args.show_opengl)
    dev = CollectDevice(args.device, valid_t_dof=env.env.gripper.valid_t_dof, valid_r_dof=env.env.gripper.valid_r_dof,
                        output_grip=env.grip_in_action, action_multiplier=args.action_multiplier)
    if env.env.gripper.control_method == 'dp' and dev.dev_type == 'keyboard':
        dev.action_multiplier = .003
    if args.device == 'vr':
        dev.dev.vel_pid.Kp = 5.0

# handle images in obs
obs_is_dict = False
if type(env.observation_space) == gym.spaces.dict.Dict:
    obs_is_dict = True
    img_traj_data = []
    depth_traj_data = []

env.seed()

data_dir = os.path.join(args.directory, args.demo_name)

# create and/or read existing dataset -- if existing, user must ensure dataset matches env
act_dim = env.action_space.shape[0]
if obs_is_dict:
    obs_shape = env.observation_space.spaces['obs'].shape[0]
else:
    obs_shape = env.observation_space.shape[0]
ds = img_depth_dataset.Dataset(data_dir, state_dim=obs_shape, act_dim=act_dim)

# collection variables
status_dict = {'record': False, 'num_demos': 0,
               't': 0, 'success': False}
traj_data = []
data = []
traj_lens = []
ts = 0
dataset_total_ts = 0
fr = 0
fr_since_collect = args.collect_interval
if ros_env:
    time_per_frame = 1 / env._control_freq
    args.enforce_real_time = False
else:
    time_per_frame = env.env._time_step * 10  # always 10 substeps in sim
ep_r = 0
old_data = None

# load file if it exists
traj_lens_filename = data_dir + '/traj_lens.npy'
if not obs_is_dict:
    os.makedirs(data_dir, exist_ok=True)
    np_filename = data_dir + '/data.npy'
    if os.path.exists(np_filename):
        traj_lens = np.load(traj_lens_filename).tolist()
        old_data = np.load(np_filename)
        dataset_total_ts = np.sum(traj_lens)

cur_base_pos = None
cur_pos = None

if args.ros_env_vr_teleop_help:
    env.set_reset_teleop_complete()
obs = env.reset()

while(True):
    frame_start = time.time()

    cancel, save, start, reset, delete, success_fb_suc, success_fb_fail = dev.update_and_get_state()
    if delete:
        print('Deleting previous trajectory')
        if obs_is_dict:
            ds.remove_last_traj()
        else:
            if len(data) > 0:
                data.pop()
                traj_lens.pop()

    # hardcoded differences between ros envs and old sim envs
    if ros_env:
        if env._control_type == 'delta_tool':
            if args.device == 'vr':
                dev.force_feedback(env)
                cur_pos = env.get_cur_base_tool_pose()
            act = dev.get_ee_vel_action(cur_pos, vr_p_mult=1.0)

    else:
        if env._control_type == 'v' or env._control_type == 'dp':
            if args.device == 'vr':
                obs_dict = env.env.gripper.receive_observation(ref_frame_pose=env.env.vel_control_frame,
                                                              ref_frame_vel=env.env.vel_ref_frame)
                cur_pos = np.concatenate([obs_dict['pos'], obs_dict['orient']])
            act = dev.get_ee_vel_action(cur_pos)
        elif env._control_type == 'p':
            #TODO get ee and base pose
            act = dev.get_ee_pos_action(None, None)

    # get done success/failure -- don't allow this to be pressed if not recording
    if ros_env and (env.success_causes_done or env.failure_causes_done) and dev.recording:
        if success_fb_suc:
            env.set_done(True)
        elif success_fb_fail:
            env.set_done(False)

    next_obs, rew, done, info = env.step(act)

    # allow setting the success bit with vr remote
    if done and ros_env and env._reset_teleop_available and env._success_feedback_available \
        and not (env.done_success or env.done_failure):
        if args.device == 'vr':
            dev.dev.trigger_haptic()
            dev.dev.reset_ref_poses()
        success_fb_suc, success_fb_fail = False, False
        while not (success_fb_suc or success_fb_fail):
            _, _, _, _, _, success_fb_suc, success_fb_fail = dev.update_and_get_state()
            time.sleep(.02)
        if success_fb_suc:
            info['done_success'] = True

    if dev.recording:
        ep_r += rew
    if info['done_success']:
        status_dict['success'] = True
    env.render()
    if dev.recording:
        if fr_since_collect == args.collect_interval:
            if obs_is_dict:
                # useful to ensure Q function is stationary on timeout
                # so even if env gives "done", this mask will be NOT_DONE on timeout
                if not (done or ts + 1 == env._max_episode_steps):
                    done_mask = Mask.NOT_DONE.value
                else:
                    done_mask = Mask.DONE.value
                traj_data.append(np.concatenate((obs['obs'], np.array(act).flatten(),
                                                 np.array([rew]), np.array([done_mask]), np.array([done]) )))
                img_traj_data.append(obs['img'])
                depth_traj_data.append(obs['depth'])
            else:
                traj_data.append(np.concatenate((obs, np.array(act).flatten(), np.array([rew]))))
            fr_since_collect = 0
        else:
            fr_since_collect += 1
        ts += 1
    else:
        env.ep_timesteps = 0  # don't start incrementing env time until recording starts
        ts = 0

    if cancel:
        traj_data = []
        if obs_is_dict:
            img_traj_data = []
            depth_traj_data = []

    # otherwise, save is defined by device pressing reset
    if args.save_on_success_only:
        if done and dev.recording and info['done_success']:
            save = True
        else:
            save = False

    if save:
        if obs_is_dict:
            # add one more observation as final obs, with no action
            traj_data.append(np.concatenate([
                next_obs['obs'], np.zeros_like(act).flatten(), np.array([0]), np.array([done_mask]), np.array([done])
            ]))
            img_traj_data.append(next_obs['img'])
            depth_traj_data.append(next_obs['depth'])

            ds.append_traj_data_lists(traj_data, img_traj_data, depth_traj_data, final_obs_included=True)
        else:
            data.append(np.array(traj_data))
            traj_lens.append(ts)

            if old_data is None:
                np.save(np_filename, np.vstack(data))
                np.save(traj_lens_filename, np.vstack(traj_lens))
            else:
                np.save(np_filename, np.concatenate([old_data, np.vstack(data)]))
                np.save(traj_lens_filename, np.vstack(traj_lens))
        dev.recording = False
        traj_data = []
        if obs_is_dict:
            img_traj_data = []
            depth_traj_data = []

    if reset or done:
        if args.device == 'vr':
            dev.dev.reset_ref_poses()
        print('Episode reward: %4.3f' % ep_r)
        obs = env.reset()

        # reset with teleop for convenience
        if ros_env and env._reset_teleop_available:
            reset = False
            while not reset:
                cancel, _, start, reset, delete, _, _ = dev.update_and_get_state()
                if env._control_type == 'delta_tool':
                    if args.device == 'vr':
                        cur_pos = env.tf_base_tool.as_pos_quat()
                        act = dev.get_ee_vel_action(cur_pos, vr_p_mult=1.0)
                        dev.force_feedback(env)
                    else:
                        act = dev.get_ee_vel_action(vr_p_mult=1.0)
                    env.step(act, reset_teleop_step=True)
                    env.ep_timesteps = 0  # so we don't hit done during reset
            env.set_reset_teleop_complete()
            if args.device == 'vr':
                dev.dev.reset_ref_poses()
            obs = env.reset()
            if args.device == 'vr':
                dev.dev.reset_ref_poses()
                dev.dev.trigger_haptic()  #signal next ep ready to run

        fr_since_collect = args.collect_interval
        traj_data = []
        if obs_is_dict:
            img_traj_data = []
            depth_traj_data = []
        ts = 0
        status_dict['success'] = False
        ep_r = 0

    status_dict['record'] = dev.recording
    if obs_is_dict:
        status_dict['num_demos'] = len(ds.data['traj_lens'])
    else:
        if traj_lens is not None:
            status_dict['num_demos'] = len(traj_lens)
    status_dict['t'] = len(traj_data)
    if fr % 10 == 0:
        print(status_dict, "Reward: %3.3f" % rew)
    fr += 1

    frame_time = time.time() - frame_start
    if args.enforce_real_time:
        leftover_time = time_per_frame - frame_time
        if leftover_time > 0:
            time.sleep(leftover_time)

    obs = next_obs