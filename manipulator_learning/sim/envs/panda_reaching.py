from gym import spaces
import numpy as np

from manipulator_learning.sim.envs.thing_reaching import ThingReachingGeneric


class PandaReachXYZState(ThingReachingGeneric):
    def __init__(self, max_real_time=6, n_substeps=10, dense_reward=True, action_multiplier=0.1, **kwargs):
        self.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,))

        CONFIG = dict(
            block_random_lim=[[.35, .35]],
            init_block_pos=[[-.025, -.05]],
            block_style='small',
            init_gripper_pose=[[0.0, .5, .25], [np.pi, 0, 0]],
            robot_base_ws_cam_tf=((-.4, .6, .3), (-2, 0, -1.85)),
            pos_limits=[[.85, -.35, .655], [1.15, -0.05, 0.8]],
            valid_r_dofs=[0, 0, 0]
        )

        super().__init__('reaching_xyz', False, dense_reward, 'w',
                         state_data=('pos', 'obj_pos', 'obj_rot_z'),
                         max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier, reach_radius=.1, robot='panda',
                         limits_cause_failure=False, failure_causes_done=False, success_causes_done=False,
                         control_frame='b', **CONFIG, **kwargs)
