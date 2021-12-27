import tensorflow as tf
import copy
import numpy as np
import math
import time
from timeit import default_timer as timer
from manipulator_learning.learning.data.tf.img_replay_buffer import ImgReplayBufferDisk
from manipulator_learning.learning.agents.tf.ensemble_actors import EnsembleActor
from manipulator_learning.learning.agents.tf.cnn_models import CNNRGBDepthCombinedState
from manipulator_learning.learning.utils.zfilter import ZFilter
from manipulator_learning.learning.utils.tf.general import set_training_seed


def calculate_bc_loss_tf(
        policy,
        obs,
        actions,
        actor_ensemble=False,
        loss_func='mse',
        time_data=None
):
    forward_start = timer()
    if policy.num_ensemble > 1:
        # p_act comes out as num_ensemble * mb_size * act_size, actions is mb_size * (num_ensemble * act_size)
        p_act = policy(obs, correct_input_size_for_ensemble=True)
        if time_data is not None: time_data['forward'].append(timer() - forward_start)
        actions = tf.transpose(tf.reshape(actions, [p_act.shape[1], p_act.shape[0], p_act.shape[2]]), [1, 0, 2])
    else:
        p_act = policy(obs)
        if time_data is not None: time_data['forward'].append(timer() - forward_start)
    if actor_ensemble:
        actions = tf.reshape(tf.tile(actions, [p_act.shape[0], 1]),
                             [p_act.shape[0], p_act.shape[1], p_act.shape[2]])

    if loss_func == 'mse':
        mse = tf.keras.losses.MeanSquaredError()
        return mse(actions, p_act)
    elif loss_func == 'huber':
        huber = tf.keras.losses.Huber()
        return huber(actions, p_act)
    else:
        return NotImplementedError("Not implemented for loss function \'%s\'" % loss_func)


def calc_loss_and_apply_grads(mode, policy, optimizer, b_obs, e_act, actor_ensemble, loss_func, time_data=None):
    if mode == 't':
        backward_start = timer()
        with tf.GradientTape() as tape:
            loss = calculate_bc_loss_tf(policy, b_obs, e_act, actor_ensemble, loss_func, time_data=time_data)
            grads = tape.gradient(loss, policy.variables)
            optimizer.apply_gradients(zip(grads, policy.variables))
        if time_data is not None: time_data['backward'].append(timer() - backward_start)

    elif mode == 'v':
        loss = calculate_bc_loss_tf(policy, b_obs, e_act, actor_ensemble, loss_func, time_data=time_data)

    return loss


def behavior_clone_tf(
        policy,
        obs=None,
        actions=None,
        img_replay_buffer: ImgReplayBufferDisk = None,
        optimizer=None,
        max_epochs=500,
        mb_size=64,
        max_epochs_wo_best=30,
        lr=1e-3,
        valid_prop=.2,
        output_freq=20,
        loss_func='mse',
        include_depth=True,
        include_state=True,
        seed=None,
        num_trajs=None,  # only used by writer, if writer is None doesn't need to be passed
        writer: tf.summary.SummaryWriter = None,
        time_info_debug=True
):
    """
    Behavior cloning for tf model. Ensure that eager execution is enabled or that tf version >= 2.0.

    :param policy:
    :param policy_step:
    :param optimizer:
    :param obs:                  np array, N x (obs size)
    :param actions:              np array, N x (act size)
    :param max_epochs:
    :param mb_size:
    :param max_epochs_wo_best:
    :param lr:
    :param valid_prop:           proportion of validation data
    :return:
    """
    if seed is not None:
        set_training_seed(seed)
    time_data = dict(forward=[], backward=[], load_data=[])
    actor_ensemble = type(policy) == EnsembleActor  # todo this is not used for image-based policies and is not used
    combine_rgb_depth = issubclass(type(policy), CNNRGBDepthCombinedState)

    assert (obs is not None and actions is not None) or img_replay_buffer is not None, "Either input all" \
                                                                                       "obs and actions, or an instance of img_replay_buffer.ImgReplayBufferDisk"

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if img_replay_buffer is None:
        N = int(obs.shape[0])
    else:
        N = len(img_replay_buffer)

    if img_replay_buffer is not None:
        # split into train and valid
        shuf_orig_indices = copy.deepcopy(img_replay_buffer._rb_valid_indices)
        np.random.shuffle(shuf_orig_indices)
        train_replay_buffer = img_replay_buffer.get_copy(use_same_dataset_obj=True)
        valid_replay_buffer = img_replay_buffer.get_copy(use_same_dataset_obj=True)
        train_replay_buffer.update_tf_dataset_params(new_valid_indices=shuf_orig_indices[math.floor(valid_prop * N):],
                                                     new_batch_size=mb_size, repeat=1,
                                                     load_to_ram=True, load_to_gpu=False, gen_tf_dataset=False)
        valid_replay_buffer.update_tf_dataset_params(new_valid_indices=shuf_orig_indices[:math.floor(valid_prop * N)],
                                                     new_batch_size=mb_size, repeat=1,
                                                     load_to_ram=True, load_to_gpu=False, gen_tf_dataset=False)

        train_replay_buffer.gen_python_dataset(num_ensemble=policy.num_ensemble, combine_rgb_depth=combine_rgb_depth)
        valid_replay_buffer.gen_python_dataset(num_ensemble=policy.num_ensemble, combine_rgb_depth=combine_rgb_depth)

        bufs = dict(t=train_replay_buffer, v=valid_replay_buffer)
        if policy.num_ensemble > 1:
            # need to use fancy indexing instead of direct slice, since data is no longer continguous
            full_state_data_size = train_replay_buffer.ind['d'] + 1
            obs_slices = []
            act_slices = []
            for i in range(policy.num_ensemble):
                offset = i * full_state_data_size
                obs_slices.append(range(train_replay_buffer.ind['s'] + offset, train_replay_buffer.ind['a'] + offset))
                act_slices.append(range(train_replay_buffer.ind['a'] + offset, train_replay_buffer.ind['r'] + offset))
            obs_slice = np.concatenate(obs_slices)
            act_slice = np.concatenate(act_slices)
        else:
            obs_slice = slice(train_replay_buffer.ind['s'], train_replay_buffer.ind['a'])
            act_slice = slice(train_replay_buffer.ind['a'], train_replay_buffer.ind['r'])

    else:
        perm = np.arange(N)
        np.random.shuffle(perm)
        perm = dict(t=perm[math.floor(valid_prop * N):], v=perm[:math.floor(valid_prop * N)])

    lowest_valid_loss = 1e100
    epochs_wo_best = 0
    for e in range(max_epochs):
        epoch_start = timer()

        running_loss = dict(t=ZFilter(1), v=ZFilter(1))
        for mode in ['v', 't']:
            if img_replay_buffer is None:
                n_mode = len(perm[mode])
                for mb_start in range(0, n_mode, mb_size):
                    ordered_ind = slice(mb_start, mb_start + min(mb_size, n_mode - mb_start))
                    ind = perm[mode][ordered_ind]
                    tic = timer()
                    e_act = tf.Variable(actions[ind])
                    b_obs = tf.Variable(obs[ind])
                    time_data['load_data'].append(timer() - tic)
                    loss = calc_loss_and_apply_grads(mode, policy, optimizer, b_obs, e_act, actor_ensemble, loss_func,
                                                     time_data=time_data)
                    running_loss[mode](np.array([loss]))
            else:
                batch_load_start = timer()
                for batch in bufs[mode].tf_ds_mapped:
                    time_data['load_data'].append(timer() - batch_load_start)
                    obs = [batch[0]]
                    if include_depth and not combine_rgb_depth:
                        obs.append(batch[1])
                    if include_state:
                        obs.append(batch[-1][:, obs_slice])

                    # actions = tf.transpose(tf.gather_nd(tf.transpose(batch[2], [1, 0]), act_slice[:, np.newaxis]), [1, 0])
                    loss = calc_loss_and_apply_grads(mode, policy, optimizer, obs,
                                                     batch[-1][:, act_slice], actor_ensemble, loss_func,
                                                     time_data=time_data)

                    running_loss[mode](np.array([loss]))
                    batch_load_start = timer()
                if mode == 't':
                    # this is a dumb workaround to allow shuffling every epoch
                    # see https://github.com/tensorflow/tensorflow/issues/27680
                    # bufs['t'].gen_tf_dataset()
                    bufs['t'].gen_python_dataset(num_ensemble=policy.num_ensemble, combine_rgb_depth=combine_rgb_depth)
                else:
                    bufs['v'].gen_python_dataset(shuffle=False, num_ensemble=policy.num_ensemble,
                                                 combine_rgb_depth=combine_rgb_depth)

            if mode == 'v':
                if e == 0:
                    lowest_valid_loss = running_loss['v'].rs.mean
                    best_weights = policy.get_weights()
                if running_loss['v'].rs.mean < lowest_valid_loss:
                    lowest_valid_loss = running_loss['v'].rs.mean
                    best_weights = policy.get_weights()
                    epochs_wo_best = 0
                else:
                    epochs_wo_best += 1

        # print('epoch time: %3.4f' % (time.time() - epoch_start))
        # print('total batch loading time: %.3f' % cum_batch_load_time)

        if tf.equal(e % 5, 0):
            tf.summary.scalar('actor/bc_t_loss', running_loss['t'].rs.mean.item(), step=e)
            tf.summary.scalar('actor/bc_v_loss', running_loss['v'].rs.mean.item(), step=e)
        if time_info_debug:
            td = dict()
            for k in time_data.keys():
                td[k] = dict()
                td[k]['median'] = np.median(time_data[k])
                td[k]['max'] = max(time_data[k])
            print("AVG -- forward: %.6f, backward: %.6f, data load: %.6f\n"
                  "MAX -- forward: %.6f, backward: %.6f, data load: %.6f" %
                  (td['forward']['median'], td['backward']['median'], td['load_data']['median'],
                   td['forward']['max'], td['backward']['max'], td['load_data']['max']))
        if e % output_freq == 0:
            print("Epoch: %d, T loss: %.6f, V loss: %.6f, Best V: %.6f, Epoch Time: %.2f, epochs w/o best: %d" %
                  (e, running_loss['t'].rs.mean, running_loss['v'].rs.mean, lowest_valid_loss, timer() - epoch_start,
                   epochs_wo_best))
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('%d demos: training/training loss' % num_trajs, running_loss['t'].rs.mean.item(),
                                  step=e)
                tf.summary.scalar('%d demos: training/valid loss' % num_trajs, running_loss['v'].rs.mean.item(), step=e)
                tf.summary.scalar('%d demos: training/epoch time' % num_trajs, timer() - epoch_start, step=e)

        if epochs_wo_best >= max_epochs_wo_best:
            break

        tf.keras.backend.clear_session()  # without this, each epoch takes longer and longer..doesn't seem to break anything

    policy.set_weights(best_weights)
    # if img_replay_buffer is not None:
    #   img_replay_buffer.dataset.flush_ram()
