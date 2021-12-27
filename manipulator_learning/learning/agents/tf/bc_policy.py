import sys
import os
import copy

import tensorflow as tf

from manipulator_learning.learning.algorithms.bc.tf.bc import behavior_clone_tf
from manipulator_learning.learning.agents.tf.ensemble_actors import EnsembleActor
from manipulator_learning.learning.data.tf.img_replay_buffer import ImgReplayBufferDisk
from manipulator_learning.learning.utils.tf.general import set_training_seed


def behavior_clone_save_load(actor, seed, num_trajs, env_str, bc_dir, expert_replay_buffer, mb_size=64, loss_func='mse',
                             include_depth=True, include_state=True, max_epochs=200, lr=1e-3, output_freq=5,
                             writer: tf.summary.SummaryWriter = None, debug=False):
    bc_name = env_str + '_' + str(num_trajs) + '_trajs_' + str(seed)
    bc_model_dir = bc_dir + '/' + bc_name
    save_name = bc_model_dir + '/bc_model'

    if os.path.exists(bc_model_dir):
        print('Loading existing BC pre-trained weights')
        actor.load_weights(save_name)
    else:
        print('Initializing Behavior Cloning')
        obs_is_dict = type(expert_replay_buffer) == ImgReplayBufferDisk
        expert_replay_buffer.subsample_trajectories(num_trajs)
        if obs_is_dict:
            behavior_clone_tf(actor, img_replay_buffer=expert_replay_buffer, output_freq=output_freq,
                              max_epochs_wo_best=30,
                              lr=lr, max_epochs=max_epochs, mb_size=mb_size, loss_func=loss_func,
                              include_depth=include_depth,
                              include_state=include_state, seed=seed, num_trajs=num_trajs, writer=writer,
                              time_info_debug=debug)
        else:
            obs, actions, _, _, _, _ = expert_replay_buffer.get_data_numpy()
            behavior_clone_tf(actor, obs, actions, lr=.001, max_epochs_wo_best=50, valid_prop=.3, max_epochs=1000,
                              mb_size=mb_size, loss_func=loss_func, time_info_debug=debug)
        actor.save_weights(save_name)


class BCWithResidualPolicy(tf.keras.Model):
    def __init__(self, actor, bc_actor):
        super().__init__()
        self.bc_actor = bc_actor
        self.resid_actor = actor
        if type(actor) == EnsembleActor:
            self.num_ens = len(actor.actors)
        else:
            self.num_ens = 1
        self.initialized = False

    def call(self, obs):
        bc_action, resid_action = self.get_bc_and_resid_action(obs)
        return bc_action + resid_action
        # if self.initialized:
        #   bc_action, resid_action = self.get_bc_and_resid_action(obs)
        #   return bc_action + resid_action
        # else:
        #   raise RuntimeError('BCWithResidualPolicy object must have initialize_bc_actor method called before being used.')

    def initialize_bc_actor(self, seed, num_trajs, env_str, bc_dir, expert_replay_buffer):
        behavior_clone_save_load(self.bc_actor, seed, num_trajs, env_str, bc_dir, expert_replay_buffer)
        self.initialized = True

    def _get_action(self, actor, obs):
        if type(actor) == EnsembleActor:
            action = actor.get_action(obs)
        else:
            action = actor(obs)
        return action

    def get_bc_and_resid_action(self, obs):
        bc_action = self._get_action(self.bc_actor, obs)
        resid_action = self._get_action(self.resid_actor, obs)
        return bc_action, resid_action

    def get_full_action_and_resid_action(self, obs):
        bc_action, resid_action = self.get_bc_and_resid_action(obs)
        return bc_action + resid_action, resid_action
