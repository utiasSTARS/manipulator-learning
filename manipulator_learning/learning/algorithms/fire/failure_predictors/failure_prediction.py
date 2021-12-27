import collections
import numpy as np

ObsAct = collections.namedtuple('ObsAct', ('obs', 'action'))
ObsActExp = collections.namedtuple('ObsActExp', ('obs', 'action', 'exp'))


class FailurePredictor:
    def __init__(self,
                 q_model,
                 q_type='td3',
                 k_prev=3,
                 gap_between_k_prev=1,
                 req_consec_under=3,
                 backend='tf',
                 q_model_step=None,
                 use_delta_q=True,
                 discrim=None):

        self.q_model = q_model
        self.q_model_step = q_model_step
        self._use_delta_q = use_delta_q
        self._last_q = -1e100
        self.req_consec_under = req_consec_under
        self.q_type = q_type
        self.k_prev = k_prev  # how many precorrective sa pairs to add per each transition to expert data
        self.gap_between_k_prev = gap_between_k_prev
        self.backend = backend
        self.pre_corrective_pairs = []  # list containing ObsAct(o, a)
        self.episode_pairs = []  # list containg ObsActExp(o, a , e)
        self.q_values = np.array([])
        self.max_q = -1e100
        self.max_q_index = None
        self.tolerance = 0.01  # adjust the sensitivity, higher means less likely to indicate failure
        self._discrim = discrim
        self._num_consec_under = 0

    def _get_q_output(self, obs, action):
        if self.q_type == 'td3':
            if len(obs.shape) == 1:
                out, _ = self.q_model([obs], [action])
            else:
                out, _ = self.q_model(obs, action)
        return out

    def _get_batch(self):
        batch = ObsAct(*zip(*self.pre_corrective_pairs))
        return self._prep_batch_for_model(batch)

    def _get_batch_delta(self):
        batch_cur = ObsAct(*zip(*list(zip(*self.pre_corrective_pairs))[0]))
        batch_prev = ObsAct(*zip(*list(zip(*self.pre_corrective_pairs))[1]))
        return self._prep_batch_for_model(batch_cur), self._prep_batch_for_model(batch_prev)

    def _prep_obs_act_for_model(self, obs, action):
        if self.backend == 'tf':
            from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe
            obs = contrib_eager_python_tfe.Variable(obs.astype('float32'))
            action = contrib_eager_python_tfe.Variable(action.astype('float32'))
        else:
            return NotImplementedError("Failure Predictor not implemented for backend %s" % self.backend)

        return obs, action

    def _prep_batch_for_model(self, batch):
        """
        Prepares batch of observations and actions for running through q model.
        :param batch: Should be an ObsAct containing all observations in obs and all actions in action
        :return:
        """
        return self._prep_obs_act_for_model(np.stack(batch.obs).astype('float32'),
                                            np.stack(batch.action).astype('float32'))

    def update_q_values(self):
        # user calls this after updating q model and before starting next episode
        if len(self.pre_corrective_pairs) > 0:
            if self._use_delta_q:
                self._update_q_values_delta()
            else:
                obs, action = self._get_batch()
                self.q_values = np.array(self._get_q_output(obs, action))
            self._update_max()

    def _update_q_values_delta(self):
        [obs, action], [prev_obs, prev_action] = self._get_batch_delta()
        q_values_cur = np.array(self._get_q_output(obs, action))
        q_values_prev = np.array(self._get_q_output(prev_obs, prev_action))
        self.q_values = q_values_cur - q_values_prev

    def _update_max(self):
        if len(self.pre_corrective_pairs) > 0:
            self.max_q = self.q_values.max()
            if self.max_q > 0:
                pass
                # import ipdb; ipdb.set_trace()
            if self._use_delta_q:
                self.max_q = min(self.max_q, 0)  # no threshold should be allowed to be above 0
            self.max_q_index = self.q_values.argmax()
        else:
            self.max_q = -1e100
            self.max_q_index = None

    def _push_back_pc(self, obs, action):
        self.pre_corrective_pairs.append(ObsAct(obs, action))

    def _push_back_pc_delta(self, obs, action, prev_obs, prev_action):
        self.pre_corrective_pairs.append([ObsAct(obs, action), ObsAct(prev_obs, prev_action)])

    def push_back_ep(self, obs, action, exp):
        # user should call this on every step of an episode
        # exp is a mask indicating whether action was expert or not, 1 for exp, 0 for not
        self.episode_pairs.append(ObsActExp(obs, action, exp))

    def update_pc_list_from_ep(self):
        # user calls this at the end of an episode
        test_REMOVE = False
        if len(self.episode_pairs) > 0:
            batch = ObsActExp(*zip(*self.episode_pairs))
            obs = np.stack(batch.obs)
            action = np.stack(batch.action)
            exp = np.stack(batch.exp)
            for i in range(obs.shape[0] - 1):
                if exp[i + 1] and not exp[i]:  # means we're at a transition to expert data
                    test_REMOVE = True
                    for k in range(self.k_prev):
                        ind = i - k * self.gap_between_k_prev
                        if exp[ind] or ind < 0: break
                        if self._use_delta_q and (exp[ind - 1] or ind - 1 < 0): break
                        if self._use_delta_q:
                            self._push_back_pc_delta(obs[ind], action[ind], obs[ind - 1], action[ind - 1])
                        else:
                            self._push_back_pc(obs[ind], action[ind])

            if test_REMOVE:
                pass
                # import ipdb; ipdb.set_trace()
            self.episode_pairs = []
            self._last_q = -1e100

    def worse_than_best(self, obs, action):
        # called by user to check if current o-a pair predicts failure
        # if self.q_model_step is not None and self.q_model_step < 300:
        #   return False
        prepped = self._prep_batch_for_model(ObsAct(obs, action))
        q_pred = np.array(self._get_q_output(prepped[0], prepped[1]))
        if self._use_delta_q:
            pred_delta_q = q_pred - self._last_q
            self._last_q = q_pred
            under_thresh = pred_delta_q < self.max_q - self.tolerance
        else:
            under_thresh = q_pred < self.max_q - self.tolerance

        self._num_consec_under = self._num_consec_under + 1 if under_thresh else 0
        if under_thresh:
            print('delq: %2.3f, max q: %2.3f, num_consec: %d' % (
            pred_delta_q, self.max_q - self.tolerance, self._num_consec_under))
        if self._use_delta_q:
            return under_thresh and self._num_consec_under >= self.req_consec_under, pred_delta_q, self.max_q
        else:
            return under_thresh and self._num_consec_under >= self.req_consec_under, q_pred, self.max_q

    def remove_best(self):
        if len(self.pre_corrective_pairs) > 0:
            # called by user if worse_than_best gave incorrect output
            self.pre_corrective_pairs.pop(self.max_q_index)
            self.q_values = np.delete(self.q_values, self.max_q_index)
            self._update_max()
