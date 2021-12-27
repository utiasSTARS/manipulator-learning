import os
import pickle
import numpy as np
import shutil
from typing import Dict
from collections import OrderedDict
import h5py

FP_STATUS_TAB = dict(tp=0, fp=1, tn=2, fn=3, tp_late=4, exp=5)


class DataRecorder:
    def __init__(self, save_dir,
                 per_ep_group_keys=('avg_suc', 'avg_ret'),
                 per_episode_keys=('success', 'ep_return', 'autonomous_success'),
                 per_timestep_keys=('reward', 'fp_raw_d_output', 'fp_status', 'int_exp_in_control',
                                    'in_exp_rb', 'rb_index', 'actor_variance')
                 ):
        """ Stores data during experimental runs in dicts, allowing for saving in hdf5 tables."""
        per_episode_defaults = ['ep_num', 'total_timesteps']
        per_timestep_defaults = ['ep_num', 'total_timesteps', 'timesteps']
        self.per_ep_group_data = OrderedDict([k, []] for k in per_ep_group_keys)
        self.per_episode_data = OrderedDict([(k, []) for k in [*per_episode_defaults, *per_episode_keys]])
        self.per_timestep_data = OrderedDict([(k, []) for k in [*per_timestep_defaults, *per_timestep_keys]])
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # also keep track of internal data to not clutter up code elsewhere
        self.internal_data = dict(ep_num=0, total_timesteps=0)

    def append_per_episode_data(self, data: Dict):
        # if data in internal data, add it to the dict
        for k in self.per_episode_data.keys():
            if k not in data:
                if k in self.internal_data.keys():
                    data[k] = self.internal_data[k]
                else:
                    data[k] = np.nan

            self.per_episode_data[k].append(data[k])

    def append_per_ep_group_data(self, data: Dict):
        for k in self.per_ep_group_data.keys():
            if k not in data:
                data[k] = np.nan
            self.per_ep_group_data[k].append(data[k])

    def append_per_timestep_data(self, dict_of_lists: Dict):
        length = len(dict_of_lists[list(dict_of_lists.keys())[0]])
        for k in self.per_timestep_data.keys():
            if k not in dict_of_lists:
                if k in self.internal_data.keys():
                    dict_of_lists[k] = [self.internal_data[k]] * length
                else:
                    dict_of_lists[k] = [np.nan] * length

            self.per_timestep_data[k].extend(dict_of_lists[k])

    # @classmethod
    # def load_from_np(cls, filepath, new_save_dir=None):
    #   """ Load recorder from existing saved data. If no new save dir, assumed to be same as existing file. """
    #   data = np.load(filepath)
    #   if new_save_dir is None:
    #     save_dir = os.path.dirname(os.path.abspath(filepath))
    #   else:
    #     save_dir = new_save_dir
    #   recorder = DataRecorder(save_dir)

    def save_as_pickle(self, filename):
        """ Save recorder object to pickle file. """
        pickle.dump(self, open(self.save_dir + '/' + filename + '.pkl', 'wb'))

    @classmethod
    def load_from_pickle(cls, filepath):
        """ Load recorder object from pickle file. """
        rec = pickle.load(open(filepath, 'rb'))
        return rec

    def save_data_as_np(self, filename):
        """ Save data to file, changing all data objects to np arrays. """
        save_name = self.save_dir + '/' + filename + '.npz'
        swap_name = self.save_dir + '/' + filename + '_swp.npz'
        np.savez_compressed(swap_name, **self.get_data_as_arrays())
        shutil.copy(swap_name, save_name)
        os.remove(swap_name)

    def save_data_as_hdf5(self, filename):
        save_name = self.save_dir + '/' + filename + '.hdf5'
        swap_name = self.save_dir + '/' + filename + '_swp.hdf5'

        f = h5py.File(swap_name, 'w')
        np_dict = self.get_data_as_arrays()
        if 'per_episode' in np_dict.keys():
            f.create_dataset('per_episode', data=np_dict['per_episode'], compression='gzip')
            f.attrs.create('per_episode_columns', data=np_dict['per_episode_columns'].tolist())
        if 'per_timestep' in np_dict.keys():
            f.create_dataset('per_timestep', data=np_dict['per_timestep'], compression='gzip')
            f.attrs.create('per_timestep_columns', data=np_dict['per_timestep_columns'].tolist())
        if 'per_ep_group' in np_dict.keys():
            f.create_dataset('per_ep_group', data=np_dict['per_ep_group'], compression='gzip')
            f.attrs.create('per_ep_group_columns', data=np_dict['per_ep_group_columns'].tolist())
        f.close()
        shutil.copy(swap_name, save_name)
        os.remove(swap_name)

    def save_all(self, filename):
        self.save_data_as_hdf5(filename)
        self.save_data_as_np(filename)
        self.save_as_pickle(filename)

    def get_data_as_arrays(self):
        """ Return data as dict of numpy arrays """
        np_dict = dict()
        for data_dict, key_str in zip([self.per_episode_data, self.per_timestep_data, self.per_ep_group_data],
                                      ['per_episode', 'per_timestep', 'per_ep_group']):
            # num_rows = len(data_dict['ep_num'])
            num_rows = len(data_dict[list(data_dict.keys())[0]])
            if num_rows > 0:
                col_sizes = []
                for k in data_dict.keys():
                    dat_ex = data_dict[k][0]
                    if hasattr(dat_ex, '__len__'):
                        col_sizes.append(len(dat_ex))
                    else:
                        col_sizes.append(1)
                col_starts = np.cumsum(col_sizes)[:-1]
                col_starts = np.concatenate([[0], col_starts])
                num_cols = sum(col_sizes)
                full_array = np.zeros([num_rows, num_cols])
                last_start = 0
                for i, k in enumerate(data_dict):
                    # need to rearrange data to ensure it will fit in full_array
                    np_array = np.atleast_2d(data_dict[k])
                    np_array_shape = np_array.shape
                    des_first_ax = np.argwhere(np.array(np_array_shape) == num_rows)
                    if len(des_first_ax) == 2:  # only happens if single piece of data, and calling .item breaks
                        des_first_ax = 0
                    elif len(des_first_ax) == 0:
                        raise ValueError("Probably wrong amount of data in %s data_dict for key %s, "
                                         "should be %d, but shape is %s" % (key_str, k, num_rows, np_array_shape))
                    else:
                        des_first_ax = des_first_ax.item()
                    if des_first_ax != 0:
                        np_array = np.swapaxes(np_array, 0, des_first_ax)
                    full_array[:, last_start:(last_start + col_sizes[i])] = np_array
                    last_start += col_sizes[i]
                np_dict[key_str] = full_array
                np_dict[key_str + '_columns'] = np.array(
                    [[k, start_i] for k, start_i in zip(data_dict.keys(), col_starts)])

        return np_dict
