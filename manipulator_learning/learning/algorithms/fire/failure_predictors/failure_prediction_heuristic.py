import collections
import numpy as np

import tensorflow as tf

import sys
import os
from manipulator_learning.learning.agents.tf.common import convert_env_obs_to_tuple

ObsAct = collections.namedtuple('ObsAct', ('obs', 'action'))


class FailurePreidctorHeuristic:
    def __init__(self,
                 discriminator,
                 d_model_step,
                 q_model=None,
                 init_non_inc_neg_d_thresh=5,
                 init_neg_d_thresh=20,
                 q_type='td3',
                 backend='tf',
                 d_model_step_delay=3000):
        self.discriminator = discriminator
        self.q_model = q_model
        self.d_model_step = d_model_step
        self.non_inc_neg_d_thresh = init_non_inc_neg_d_thresh
        self.consec_non_inc_neg_d = 0
        self.neg_d_thresh = init_neg_d_thresh
        self.consec_neg_d = 0
        self.q_type = q_type
        self.backend = backend
        self.d_model_step_delay = d_model_step_delay
        self.last_q = -1e10
        self.last_d = -1e10
        self.include_q_cond = q_model is not None
        self.first_episode_prediction_made = False
        self.statistics = dict(tp=0, fp=0, fn=0, tn=0)

    def _get_q_output(self, tf_obs, tf_action):
        if self.q_type == 'td3':
            if len(tf_obs.shape) == 1:
                out, _ = self.q_model([tf_obs], [tf_action])
            else:
                out, _ = self.q_model(tf_obs, tf_action)
        return out

    def _get_d_output(self, tf_obs, tf_action):
        if type(tf_obs[0]) == tuple:
            tf_obs = tf_obs[0]
            disc_input = (tf_obs[0], tf_obs[1], tf.concat([tf_obs[2], tf_action], -1))
            if self.discriminator.num_ensemble > 1:
                mean, std = self.discriminator.forward_mean_std(disc_input)
                return mean
            else:
                return self.discriminator(disc_input)
        else:
            if self.discriminator.num_ensemble > 1:
                mean, std = self.discriminator.forward_mean_std(tf.concat([tf_obs, tf_action], -1))
                return mean
            else:
                return self.discriminator(tf.concat([tf_obs, tf_action], -1))

    def _prep_obs_act_for_model(self, obs, action):
        if self.backend == 'tf':
            if type(obs) == dict:
                obs = convert_env_obs_to_tuple(obs)
            else:
                from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe
                obs = contrib_eager_python_tfe.Variable(obs.astype('float32'))
                action = contrib_eager_python_tfe.Variable(action.astype('float32'))
        else:
            return NotImplementedError("Failure Predictor not implemented for backend %s" % self.backend)

        return obs, action

    def get_failure_prediction(self, obs, action):
        # called by user to check if current o-a pair predicts failure
        if self.d_model_step > self.d_model_step_delay:
            tf_obs, tf_action = self._prep_obs_act_for_model(obs, action)
            if self.include_q_cond:
                q_out = self._get_q_output(tf_obs, tf_action)
                q_change = q_out - self.last_q
                self.last_q = q_out
            d_out = self._get_d_output([tf_obs], [tf_action])
            d_change = d_out - self.last_d
            self.last_d = d_out

            if d_out < 0:
                self.consec_neg_d += 1
                if self.include_q_cond and q_change < 0:
                    self.consec_non_inc_neg_d += 1
                else:
                    self.consec_non_inc_neg_d = 0
            else:
                self.consec_neg_d = 0
                self.consec_non_inc_neg_d = 0

            thresholds_reached = dict(
                consec_non_inc=False,
                consec_neg_d=False
            )
            if self.consec_non_inc_neg_d >= self.non_inc_neg_d_thresh:
                thresholds_reached['consec_non_inc'] = True
            if self.consec_neg_d >= self.neg_d_thresh:
                thresholds_reached['consec_neg_d'] = True
            if thresholds_reached['consec_non_inc'] or thresholds_reached['consec_neg_d']:
                return True, thresholds_reached
            else:
                return False, thresholds_reached
        else:
            return False, None

    def update_thresh_false_positive(self, thresholds_reached, delta_non_inc=1, delta_neg_d=1):
        # called by user to modify thresholds based on output from get_failure_prediction, if false positive
        # failure prediction (so thresholds are increased)
        if not thresholds_reached['consec_non_inc']:
            delta_non_inc = 0
        if not thresholds_reached['consec_neg_d']:
            delta_neg_d = 0
        self.update_consec_thresh(delta_non_inc, delta_neg_d)

    def update_consec_thresh(self, noninc_change, neg_d_change):
        self.non_inc_neg_d_thresh += noninc_change
        self.non_inc_neg_d_thresh = max(self.non_inc_neg_d_thresh, 1)
        self.neg_d_thresh += neg_d_change
        # self.neg_d_thresh = max(self.neg_d_thresh, 1)
        self.neg_d_thresh = max(self.neg_d_thresh, self.non_inc_neg_d_thresh + 1)  # beta > alpha
        print('New thresholds -- non-inc: %d, neg-d: %d' % (self.non_inc_neg_d_thresh, self.neg_d_thresh))

    def update_consec_counts(self, noninc_change, neg_d_change):
        self.consec_non_inc_neg_d += noninc_change
        self.consec_non_inc_neg_d = max(0, self.consec_non_inc_neg_d)
        self.consec_neg_d += neg_d_change
        self.consec_neg_d = max(0, self.consec_neg_d)

    def new_ep_reset(self):
        # called by user at beginning of new episode
        self.last_q = -1e10
        self.last_d = -1e10
        self.reset_consec_counts()
        self.first_episode_prediction_made = False

    def reset_consec_counts(self):
        self.consec_neg_d = 0
        self.consec_non_inc_neg_d = 0

    def update_statistics(self, update_str):
        # called by user to update statistics
        if not self.first_episode_prediction_made and self.d_model_step > self.d_model_step_delay:
            self.first_episode_prediction_made = True
            self.statistics[update_str] += 1
