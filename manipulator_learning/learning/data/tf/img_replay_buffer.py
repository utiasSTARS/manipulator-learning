""" Implementation of a disk-based replay buffer for images """
import numpy as np
import copy
import random

import tensorflow as tf

from manipulator_learning.learning.agents.tf.common import convert_env_obs_to_tuple
import manipulator_learning.learning.data.img_depth_dataset as img_depth_dataset

# means and stds from imagenet
imgnet_means = [0.485, 0.456, 0.406]
imgnet_stds = [0.229, 0.224, 0.225]


class ImgReplayBufferRAM:
    """ A RAM based buffer for img data, expected use is as a single episode buffer to be later appended
    to a disk buffer"""

    def __init__(self):
        self.data_buf = []
        self.img_buf = []
        self.depth_buf = []
        self.valid_indices = []
        self.current_index = 0

    def __len__(self):
        return len(self.valid_indices)

    def push_back(self, obs, action, reward, mask, done, act_was_executed=False):
        # if act_was_executed is False, means that the next obs DOESN'T follow from this action,
        # so this piece of data isn't considered "valid" b/c the next obs after it can't be used to train q
        self.data_buf.append(np.concatenate([
            obs['obs'].flatten(), np.array(action).flatten(), np.array([reward]), np.array([mask]), np.array([done])]))
        self.img_buf.append(obs['img'])
        self.depth_buf.append(obs['depth'])
        if act_was_executed:
            self.valid_indices.append(self.current_index)
        self.current_index += 1

    def combine(self, other_buffer, start_index=None, end_index=None):
        raise NotImplementedError()
        # if len(other_buffer) == 0:
        #   return
        # if type(other_buffer) == ImgReplayBufferRAM:
        #   self.data_buf.extend(other_buffer.data_buf)
        #   self.img_buf.extend(other_buffer.img_buf)
        #   self.depth_buf.extend(other_buffer.depth_buf)
        #   self.valid_indices.extend(list(np.array(other_buffer.valid_indices) + self.curre))
        # else:
        #   raise TypeError("other buffer must be ImgReplayBufferRAM")

    def remove_last(self):
        self.data_buf = self.data_buf[:self.valid_indices[-1] - 1]
        self.img_buf = self.data_buf[:self.valid_indices[-1] - 1]
        self.depth_buf = self.data_buf[:self.valid_indices[-1] - 1]
        self.valid_indices.pop()
        self.current_index = self.valid_indices[-1] + 1

    def flush(self):
        self.data_buf = []
        self.img_buf = []
        self.depth_buf = []
        self.valid_indices = []
        self.current_index = 0


class ImgReplayBufferDisk:
    def __init__(self, data_dir, state_dim=None, act_dim=None, batch_size=128, num_workers=8,
                 valid_indices=None, repeat=None, init_tf_dataset=False,
                 existing_dataset_obj=None, orig_dataset_obj_len=None, new_data_len=None):
        if existing_dataset_obj is not None:  # so we can save on used memory if dataset loaded to ram
            self.dataset = existing_dataset_obj
        else:
            self.dataset = img_depth_dataset.Dataset(data_dir)

        # now that the dataset object keeps track of the act dim and state dim, we don't necessarily need
        # them as arguments to create the replay buffer, but we still need to do checks
        if not (hasattr(self.dataset, 'ind') and self.dataset.ind is not None) and (
                act_dim is None or state_dim is None):
            raise AttributeError('Either state_dim and act_dim must be specified, or the dataset object'
                                 'at %s must have the data[\'column_headers\'] key which has this information' % data_dir)
        if self.dataset.ind is not None:
            if act_dim is None:
                act_dim = self.dataset.ind['r'] - self.dataset.ind['a']
            else:
                assert (self.dataset.ind['r'] - self.dataset.ind['a']) == act_dim
            if state_dim is None:
                state_dim = self.dataset.ind['a']
            else:
                assert self.dataset.ind['a'] == state_dim

        # orig_dataset_obj_len and new_data_len allow us to remove new data conveniently if we use subsample
        # trajectories with a collection of demonstrations initially. really, what we should do is remove unnecessary
        # data and save a new underlying dataset structure to a new folder, but this would take a fair bit of time to
        # implement and may be more trouble than it's worth.
        if orig_dataset_obj_len is not None:  # never want to remove any data with indices below this amount
            self.orig_dataset_obj_len = orig_dataset_obj_len
        else:
            self.orig_dataset_obj_len = len(self.dataset)
        if new_data_len is not None:
            self.new_data_len = new_data_len
        else:
            self.new_data_len = 0
        self.state_dim = state_dim  # auxiliary states, not img states
        self.act_dim = act_dim

        self.ind = dict(
            s=0, a=state_dim, r=state_dim + act_dim, m=state_dim + act_dim + 1, d=state_dim + act_dim + 2)
        if valid_indices is None:
            self._rb_valid_indices = copy.deepcopy(self.dataset.data['valid_indices'])
        else:
            self._rb_valid_indices = valid_indices
        self._num_workers = num_workers  # for tf.data.Dataset
        self._batch_size = batch_size  # for tf.data.Dataset
        self._repeat = repeat  # for tf.data.Dataset
        self._num_mb = int(np.ceil(len(self) / batch_size))
        self._tf_ds_num_ensemble = 1

        # tf data structure for fast fetching
        if init_tf_dataset:
            self.gen_tf_dataset()

    def __len__(self):
        return len(self._rb_valid_indices)

    @classmethod
    def load_from_save_data(cls, save_dict):
        return cls(**save_dict)

    def get_copy(self, use_same_dataset_obj=False, new_dataset_dir=None):
        """ Return a copy of this replay buffer (to avoid using copy.deepcopy or similar) """
        assert not (use_same_dataset_obj and new_dataset_dir is not None), "Either use existing dataset obj," \
                                                                           "or make new dataset obj w/ new dir, not both"
        if new_dataset_dir is None:
            init_dict = self.get_save_data()
            if use_same_dataset_obj:
                init_dict['data_dir'] = None
                init_dict['existing_dataset_obj'] = self.dataset
            return self.load_from_save_data(init_dict)
        else:
            new_dataset = self.dataset.new_dataset_from_indices(new_dataset_dir, self._rb_valid_indices)
            new_irb = ImgReplayBufferDisk(new_dataset.data_dir, self.state_dim, self.act_dim, self._batch_size,
                                          self._num_workers, repeat=self._repeat, existing_dataset_obj=new_dataset)
            return new_irb

    def get_save_data(self):
        """ Get the variables necessary for reloading this object, without saving expensive things like the actual
        data that is currently in RAM as part of the dataset object.

        To work properly with load_from_save_data, all members of this dict need to be valid keyword arguments for
        instantiating this class."""
        return dict(
            data_dir=self.dataset.data_dir, state_dim=self.state_dim, act_dim=self.act_dim, batch_size=self._batch_size,
            num_workers=self._num_workers, valid_indices=self._rb_valid_indices, repeat=self._repeat,
            orig_dataset_obj_len=self.orig_dataset_obj_len, new_data_len=self.new_data_len
        )

    # def combine(self, other_buffer, indices_to_combine=None):
    def combine(self, other_buffer, start_index=0, end_index=None):
        """ start_index isn't actually taken raw, but is the index of the 'valid_indices' of other_buffer """
        if len(other_buffer) == 0:
            return []

        if type(other_buffer) == ImgReplayBufferRAM:
            # if indices_to_combine is None:
            if start_index == 0 and end_index is None:
                new_valid_indices = other_buffer.valid_indices
                data_si = start_index;
                data_ei = end_index
            else:
                # append_traj_data_lists is expecting new_indices to start from 0, so ensure that is the case
                new_valid_indices = list(np.array(other_buffer.valid_indices[start_index:end_index]) -
                                         other_buffer.valid_indices[start_index])
                data_si = other_buffer.valid_indices[start_index]
                # add the final piece of 'invalid' data if it exists by taking final valid index + 2
                if end_index is None: end_index = -1
                data_ei = other_buffer.valid_indices[end_index] + 2

            new_dataset_indices = self.dataset.append_traj_data_lists(other_buffer.data_buf[data_si:data_ei],
                                                                      other_buffer.img_buf[data_si:data_ei],
                                                                      other_buffer.depth_buf[data_si:data_ei],
                                                                      new_indices=new_valid_indices)

            self._rb_valid_indices.extend(new_dataset_indices)
            self.new_data_len += len(new_dataset_indices)
        elif type(other_buffer) == ImgReplayBufferDisk:
            raise NotImplementedError('well go ahead and implement it if you need it')
        else:
            raise TypeError('other_buffer type is unrecognized')

        return new_dataset_indices

    def push_back(self, obs, action, next_obs, reward, mask, done):
        # obs and next_obs assumed to be dicts with 'obs', 'img', and 'depth' as keys
        # if this method is used, automatically push back a next obs with a dummy action
        # *****use of this method should be avoided in favour of using "combine" with a set
        # of data from a whole trajectory
        data_list = [np.concatenate([
            obs['obs'], np.array(action).flatten(), np.array([reward]), np.array([mask]), np.array([done])])]
        data_list.append(np.concatenate([
            next_obs['obs'], np.zeros_like(action).flatten(), np.array([0]), np.array([mask]), np.array([done])]))
        img_list = [obs['img'], next_obs['img']]
        depth_list = [obs['depth'], next_obs['depth']]
        new_indices = self.dataset.append_traj_data_lists(data_list, img_list, depth_list, final_obs_included=True)
        self._rb_valid_indices.append(new_indices)
        self.new_data_len += len(new_indices)

    def get_average_reward(self, num_trajs_to_calc_from=None):
        # this will only work when there are full trajectories in the dataset
        if num_trajs_to_calc_from is None:
            num_trajs_to_calc_from = len(self)
        reward = 0
        num_trajs = 0
        for i in self.dataset.data['valid_indices']:
            ts_da = self.dataset.data['state_data'][i]
            reward += ts_da[self.ind['r']]
            if ts_da[self.ind['d']]:
                num_trajs += 1
            if num_trajs >= num_trajs_to_calc_from:
                break
        return reward / num_trajs

    def subsample_trajectories(self, num_trajectories, new_dataset_obj_directory=None):
        traj_indices = []
        all_traj_indices = []

        for ind in self._rb_valid_indices:
            traj_indices.append(ind)
            if self.dataset.data['state_data'][ind, self.ind['d']]:
                all_traj_indices.append(traj_indices)
                traj_indices = []

        if len(all_traj_indices) < num_trajectories:
            raise ValueError('Not enough trajectories to subsample')
        subsampled = random.sample(all_traj_indices, num_trajectories)

        # length = len(all_traj_indices[0])
        # for i, traj in enumerate(all_traj_indices):
        #   assert len(traj) == length, "all trajectory lengths are not the same, traj %d has length %d, should have" \
        #                              "length %d" % (i, len(traj), length)

        # needed for fire if we're going to be adding data, if we're not adding any new data (e.g. for bc),
        # then no need to create a new folder with contiguous valid indices
        if new_dataset_obj_directory is not None:
            new_dataset = self.dataset.new_dataset_from_indices(new_dataset_obj_directory,
                                                                list(np.concatenate(subsampled)))
            self.__init__(new_dataset.data_dir, self.state_dim, self.act_dim, self._batch_size, self._num_workers,
                          repeat=self._repeat, existing_dataset_obj=new_dataset)
        else:
            self._rb_valid_indices = list(np.concatenate(subsampled))

        # self.gen_tf_dataset()

    def gen_tf_dataset(self):

        self.tf_ds = tf.data.Dataset.from_tensor_slices(self._rb_valid_indices)

        self.tf_ds_shuffled = self.tf_ds.shuffle(len(self))  # not sure about how slow this will get

        if self.dataset.ram_data is None and self.dataset.gpu_data is None:
            # this is ugly but it seems to work okay...thanks tensorflow
            # see https://github.com/tensorflow/tensorflow/issues/27811, https://github.com/tensorflow/tensorflow/issues/30653
            self.tf_ds_mapped = self.tf_ds_shuffled.map(lambda x, self=self: tf.py_function(
                self._get_data_python_map, [x], Tout=(tf.float32, tf.float32, tf.float32)),
                                                        num_parallel_calls=self._num_workers)
            # self._get_data_python_map, [x], Tout=(tf.float32, tf.float32, tf.float32, tf.int32)), num_parallel_calls=self._num_workers)
        elif self.dataset.ram_data is not None and self.dataset.gpu_data is None:
            self.tf_ds_mapped = self.tf_ds_shuffled.map(lambda x, self=self: tf.py_function(
                self._get_ram_data_python_map, [x], Tout=(tf.float32, tf.float32, tf.float32)),
                                                        # self._get_ram_data_python_map, [x], Tout=tf.float32),
                                                        num_parallel_calls=self._num_workers)
        elif self.dataset.gpu_data is not None:
            self.tf_ds_mapped = self.tf_ds_shuffled.map(lambda x, self=self: tf.py_function(
                self._get_gpu_data_python_map, [x], Tout=(tf.float32, tf.float32, tf.float32)),
                                                        # self._get_ram_data_python_map, [x], Tout=tf.float32),
                                                        num_parallel_calls=self._num_workers)

        # self.tf_ds_mapped = self.tf_ds_mapped.cache()  # doesn't work with GPU for some reason
        # self.tf_ds_mapped.apply(tf.data.experimental.prefetch_to_device('gpu'))
        self.tf_ds_mapped = self.tf_ds_mapped.repeat(self._repeat)
        self.tf_ds_mapped = self.tf_ds_mapped.batch(self._batch_size)
        self.tf_ds_mapped = self.tf_ds_mapped.prefetch(2)  # TODO not sure about this
        self.tf_ds_iter = iter(self.tf_ds_mapped)

    def ram_data_gen(self, shuffle=True, infinite=False, num_ensemble=1, combine_rgb_depth=False):
        # if num_ensemble > 1, allows generating all data in different order for each ensemble member,
        #   also elongating data along the channel dimension of each piece of data
        assert self.dataset.ram_data is not None, 'Must call load_to_ram before creating a ram data generator'

        data = self.dataset.ram_data
        if self.dataset.gpu_data is not None:
            data = self.dataset.gpu_data

        first_iter = True
        while first_iter or infinite:
            if shuffle:
                if num_ensemble == 1:
                    shuf = np.random.permutation(self._rb_valid_indices)
                else:
                    shuf = [np.random.permutation(self._rb_valid_indices) for i in range(num_ensemble)]
            else:
                if num_ensemble == 1:
                    shuf = self._rb_valid_indices
                else:
                    shuf = [self._rb_valid_indices] * num_ensemble
            num_mbs = int(np.ceil(len(self) / self._batch_size))
            if infinite: num_mbs -= 1  # ensures that all batches are the same size, and we just don't train on final part of data

            # set data in form needed for indexing immediately so it can be indexed quickly
            if num_ensemble > 1:
                if combine_rgb_depth:
                    shuf_img_depth, shuf_state = [], []
                    for e_inds in shuf:
                        shuf_img_depth.append(data['img'][e_inds])
                        shuf_img_depth.append(data['depth'][e_inds])
                        shuf_state.append(data['state'][e_inds])
                    shuf_img_depth = np.concatenate(shuf_img_depth, axis=-1)
                    shuf_state = np.concatenate(shuf_state, axis=-1)

                else:
                    shuf_img, shuf_depth, shuf_state = [], [], []
                    for e_inds in shuf:
                        shuf_img.append(data['img'][e_inds])
                        shuf_depth.append(data['depth'][e_inds])
                        shuf_state.append(data['state'][e_inds])
                    shuf_img = np.concatenate(shuf_img, axis=-1)
                    shuf_depth = np.concatenate(shuf_depth, axis=-1)
                    shuf_state = np.concatenate(shuf_state, axis=-1)

            for i_mb in range(num_mbs):
                d_slice = slice(int(i_mb * self._batch_size), int(min((i_mb + 1) * self._batch_size, len(self))))
                if num_ensemble == 1:
                    inds = shuf[d_slice]
                    if combine_rgb_depth:
                        yield np.concatenate([data['img'][inds], data['depth'][inds]], axis=-1), data['state'][inds]
                    else:
                        yield data['img'][inds], data['depth'][inds], data['state'][inds]
                else:
                    if combine_rgb_depth:
                        yield shuf_img_depth[d_slice], shuf_state[d_slice]
                    else:
                        yield shuf_img[d_slice], shuf_depth[d_slice], shuf_state[d_slice]
            first_iter = False

    def gen_python_dataset(self, shuffle=True, infinite=False, num_ensemble=1, combine_rgb_depth=False):
        """ Create a python generator as an alternative to a tf mapped dataset"""
        self._tf_ds_num_ensemble = num_ensemble
        self.tf_ds_mapped = self.ram_data_gen(shuffle, infinite, num_ensemble, combine_rgb_depth)

        if num_ensemble > 1:
            # set up dataset outputs using fancy indexing
            full_state_data_size = self.ind['d'] + 1
            self._ens_inds = dict(o=[], a=[], r=[], m=[], d=[])
            for i in range(num_ensemble):
                offset = i * full_state_data_size
                self._ens_inds['o'].append(range(self.ind['s'] + offset, self.ind['a'] + offset))
                self._ens_inds['a'].append(range(self.ind['a'] + offset, self.ind['r'] + offset))
                self._ens_inds['r'].append([self.ind['r'] + offset])
                self._ens_inds['m'].append([self.ind['m'] + offset])
                self._ens_inds['d'].append([self.ind['d'] + offset])
            for k in self._ens_inds.keys():
                self._ens_inds[k] = np.concatenate(self._ens_inds[k])

    def update_tf_dataset_params(self, new_batch_size=None, new_valid_indices=None, repeat=None, num_workers=None,
                                 load_to_ram=False, load_to_gpu=False, gen_tf_dataset=True):
        if new_batch_size is not None:
            self._batch_size = new_batch_size
        if new_valid_indices is not None:
            self._rb_valid_indices = new_valid_indices
        if repeat is not None:
            self._repeat = repeat
        if num_workers is not None:
            self._num_workers = num_workers
        if load_to_ram or load_to_gpu:
            self.dataset.load_to_ram(num_workers, True, True, True)
            if load_to_gpu:
                if self.dataset.gpu_data is None:
                    self.dataset.gpu_data = dict(
                        img=tf.constant(tf.image.convert_image_dtype(self.dataset.ram_data['img'], tf.float32),
                                        name='img'),
                        depth=tf.constant(np.expand_dims(self.dataset.ram_data['depth'], axis=-1), name='depth'),
                        state=tf.constant(self.dataset.ram_data['state'].astype('float32'), name='state')
                    )
                else:
                    print('data already loaded to gpu. Flush data with dataset.flush before reloading.')

        self._num_mb = int(np.ceil(len(self) / self._batch_size))

        if gen_tf_dataset:
            self.gen_tf_dataset()

    def load_to_gpu(self):
        assert self.dataset.ram_data is not None, "dataset object must be loaded to ram first"
        if self.dataset.gpu_data is not None:
            del self.dataset.gpu_data
        self.dataset.gpu_data = dict(
            img=tf.constant(tf.image.convert_image_dtype(self.dataset.ram_data['img'], tf.float32), name='img'),
            depth=tf.constant(np.expand_dims(self.dataset.ram_data['depth'], axis=-1), name='depth'),
            state=tf.constant(self.dataset.ram_data['state'].astype('float32'), name='state')
        )

    def sample_batch(self, include_next_obs=False):
        """ Return a batch from tensorflow Dataset"""
        if include_next_obs:
            return NotImplementedError('Only need this for q learning, not doing that ATM')

        # batch = next(self.tf_ds_iter)
        batch = next(self.tf_ds_mapped)
        return (self.batch_to_components(batch, self._tf_ds_num_ensemble))

    def batch_to_components(self, batch, num_ensemble=1):
        if num_ensemble == 1:
            o = (batch[0], batch[1], batch[2][:, :self.ind['a']])
            a = batch[2][:, self.ind['a']:self.ind['r']]
            r = batch[2][:, self.ind['r']]
            m = batch[2][:, self.ind['m']]
            d = batch[2][:, self.ind['d']]
        else:
            o = (batch[0], batch[1], batch[2][:, self._ens_inds['o']])
            a = batch[2][:, self._ens_inds['a']]
            r = batch[2][:, self._ens_inds['r']]
            m = batch[2][:, self._ens_inds['m']]
            d = batch[2][:, self._ens_inds['d']]
        return dict(o=o, a=a, r=r, m=m, d=d)

    def _test_python_map(self, i):
        i = i.numpy().item()
        return str(i)

    def _get_gpu_data_python_map(self, i):
        i = i.numpy().item()
        return self.dataset.gpu_data['img'][i], self.dataset.gpu_data['depth'][i], self.dataset.gpu_data['state'][i]

    def _get_ram_data_python_map(self, i):
        i = i.numpy().item()
        # img_float = tf.image.convert_image_dtype(self.dataset.ram_data['img'][i], tf.float32)
        img = self.dataset.ram_data['img'][i]
        # depth = np.expand_dims(self.dataset.ram_data['depth'][i], axis=-1)
        depth = self.dataset.ram_data['depth'][i]
        state = self.dataset.ram_data['state'][i]
        return img, depth, state

    def _get_data_python_map(self, i):
        # todo no image normailzation, decide if necessary
        i = i.numpy().item()
        img_file_str = self.dataset.get_img_file(i, True)
        img = tf.io.read_file(img_file_str)
        img_float = self._img_decode(img)
        depth = np.expand_dims(np.load(self.dataset.get_depth_file(i, True)), axis=-1)
        state = self.dataset.data['state_data'][i]
        # return img_float, depth, state, i
        return img_float, depth, state
        # return img_float

    def _img_decode(self, img):
        img = tf.image.decode_png(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def tuple_batch_to_d_input(self, batch):
        """ Convert a batch tuple (img, depth, state) to an input for a discriminator """
        dict = self.batch_to_components(batch)
        obs = dict['o']
        action = dict['a']
        state_inputs = obs[2]
        state_inputs = tf.concat([state_inputs, action], -1)
        return (obs[0], obs[1], state_inputs)

    def obs_act_to_d_input(self, obs, action):
        obs = convert_env_obs_to_tuple(obs)
        return (obs[0], obs[1], tf.concat([obs[2], [action]], -1))
