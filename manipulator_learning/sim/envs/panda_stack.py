from gym import spaces
import numpy as np

from manipulator_learning.sim.envs.thing_stack import ThingStackGeneric


# this one is always blue on green for success
class PandaStackXYZState(ThingStackGeneric):
    def __init__(self, max_real_time=18, n_substeps=10, dense_reward=True, action_multiplier=0.1, **kwargs):
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (19,), dtype=np.float32)

        CONFIG = dict(
            block_style='small',
            block_random_lim=[[0.15, 0.15], [0.15, 0.15]],
            init_block_pos=[[-.025, -.05], [-.025, -.05]],
            init_gripper_random_lim=[.15, .15, .06, 0., 0., 0.],
            init_gripper_pose=[[0.0, .5, .25], [np.pi, 0, 0]],
            valid_r_dofs=[0, 0, 0],
            robot_base_ws_cam_tf=((-.4, .6, .3), (-2, 0, -1.85)),
            pos_limits=[[.85, -.35, .655], [1.15, -0.05, 0.8]]
        )

        super().__init__('stack_2_same_xyz', False, dense_reward, 'w',
                         state_data=('pos', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot_z'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier, robot='panda',
                         limits_cause_failure=False, failure_causes_done=False, success_causes_done=False,
                         control_frame='b', **CONFIG, **kwargs)
