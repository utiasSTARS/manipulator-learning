import numpy as np
import os
from PIL import Image
from multiprocessing import Pool
from itertools import repeat
import shutil

class Dataset:
    def __init__(self, data_dir, np_filename='data.npz', img_type='.png', imgs_per_folder=1000,
                 state_dim=None, act_dim=None, reward_dim=1):
        # valid_indices are valid to sample if looking to get a random sample. this way,
        # we can always just grab the the obs and action from the current index, and the obs from the
        # next index to get a full transition, without having to store extra copies of next observations.
        # observations at an "invalid index" are only to be used as "next observations", and the action
        # at these indices should be considered garbage!!!
        #
        # if state_dim and act_dim are not None, just means that column header data is saved as well
        self.total_ts = 0
        self.total_ts_including_last_obs = 0
        self.data_dir = data_dir
        self.data_file = self.data_dir + '/' + np_filename
        self.data = dict(
            state_data=None, traj_lens=[], traj_lens_including_last_obs=[], valid_indices=[]
        )
        if state_dim is not None and act_dim is not None:
            self.ind = dict(
                s=0, a=state_dim, r=state_dim + act_dim, m=state_dim + act_dim + reward_dim,
                d=state_dim + act_dim + reward_dim + 1)
            self.data['column_headers'] = np.array(self.ind)
        else:
            self.ind = None
            self.data['column_headers'] = np.array(None)
        self.img_dir = data_dir + '/img'
        self.depth_dir = data_dir + '/depth'
        self.img_type = img_type
        self.img_fol_str_len = 5  # up to 1e8 images if 1000 imgs per folder
        self.img_str_len = len(str(imgs_per_folder - 1))
        self.imgs_per_folder = imgs_per_folder

        if os.path.exists(self.data_file):
            file_dict = np.load(self.data_file, allow_pickle=True)
            self.data = {key: file_dict[key] for key in file_dict.files}
            if 'column_headers' not in self.data:
                self.data['column_headers'] = np.array(None)
            for k in self.data:
                if k != 'column_headers':
                    self.data[k] = list(self.data[k])

            self.data['state_data'] = np.array(self.data['state_data'])
            self.data['valid_indices'] = list(self.data['valid_indices'])
            self.ind = self.data['column_headers'].item()
            self.total_ts = sum(self.data['traj_lens'])
            self.total_ts_including_last_obs = sum(self.data['traj_lens_including_last_obs'])
            print('TOTAL TS: ', self.total_ts)

        self.ram_data = None
        self.gpu_data = None  # calling class can overwrite with gpu loaded data from torch or tf
        self._loaded_ram_indices = np.array([])

    def __len__(self):
        return self.total_ts

    def append_traj_data_lists(self, data_list, img_list, depth_list, final_obs_included=False,
                               new_indices=None):
        # if appending single piece of new data, needs to still be a len 1 list
        # if new_indices is not None, it is assumed to be a list-like set of integers
        # starting from 0 (e.g. if there are a number of invalid indices in various places)
        assert len(data_list) == len(img_list) == len(depth_list)

        # append new data to internal file
        if self.data['state_data'] is not None:
            assert self.data['state_data'].shape[1:] == data_list[0].shape
            self.data['state_data'] = np.concatenate([self.data['state_data'], np.array(data_list)])
        else:  # corresponds to first save
            os.makedirs(self.data_dir)
            os.makedirs(self.img_dir)
            os.makedirs(self.depth_dir)
            self.data['state_data'] = np.array(data_list)

        if new_indices is not None:
            number_new_data = len(new_indices)
        else:
            if final_obs_included:
                number_new_data = len(data_list) - 1
            else:
                number_new_data = len(data_list)
        self.data['traj_lens'].append(number_new_data)
        self.data['traj_lens_including_last_obs'].append(len(data_list))

        # save new imgs to disk
        for i in range(len(img_list)):
            im = Image.fromarray(img_list[i])
            im.save(self.get_img_file(self.total_ts_including_last_obs + i, load=False))
            np.save(self.get_depth_file(self.total_ts_including_last_obs + i, load=False), depth_list[i])

        if new_indices is not None:
            ds_new_indices = np.array(new_indices) + self.total_ts_including_last_obs
        else:
            ds_new_indices = range(self.total_ts_including_last_obs, self.total_ts_including_last_obs + number_new_data)
        self.data['valid_indices'].extend(ds_new_indices)
        self.total_ts += number_new_data
        self.total_ts_including_last_obs += len(data_list)

        # save new data to disk
        swap_name = self.data_file.split('.npz')[0] + '_swp.npz'  # attempt to prevent catastrophic data loss
        np.savez_compressed(swap_name, **self.data)
        shutil.copy(swap_name, self.data_file)
        # np.savez_compressed(self.data_file, **self.data)

        return list(ds_new_indices)

    def remove_last_traj(self):
        if len(self.data['traj_lens']) == 0:
            print('No trajs to remove')
            return

        # update internal data file
        last_i_to_keep = self.total_ts_including_last_obs - self.data['traj_lens_including_last_obs'][-1]
        self.data['state_data'] = self.data['state_data'][:last_i_to_keep]
        last_traj_len = self.data['traj_lens'][-1]
        last_traj_len_including_last_obs = self.data['traj_lens_including_last_obs'][-1]
        self.data['traj_lens'].pop()
        self.data['traj_lens_including_last_obs'].pop()
        self.data['valid_indices'] = self.data['valid_indices'][:-last_traj_len]

        for i in range(last_i_to_keep, self.total_ts_including_last_obs):
            os.remove(self.get_img_file(i))
            os.remove(self.get_depth_file(i))

        self.total_ts -= last_traj_len
        self.total_ts_including_last_obs -= last_traj_len_including_last_obs

        # overwrite data file on disk
        swap_name = self.data_file.split('.npz')[0] + '_swp.npz'  # attempt to prevent catastrophic data loss
        np.savez_compressed(swap_name, **self.data)
        shutil.copy(swap_name, self.data_file)
        # np.savez_compressed(self.data_file, **self.data)

    def remove_trajs(self, traj_indices, new_dataset_dir):
        """ remove trajs from a set of indices -- indices should be a list (even if it's just one)
        new_dataset_dir must be provided to avoid accidentally deleting old data. """
        vi_sorted = np.sort(self.data['valid_indices'])
        streaks = np.split(vi_sorted, np.where(np.diff(vi_sorted) != 1)[0] + 1)
        for index in sorted(traj_indices, reverse=True):
            del streaks[index]
        new_valid_indices = np.concatenate(streaks)
        print("Generating new dataset at %s with indices at %s removed" % (new_dataset_dir, traj_indices))
        self.new_dataset_from_indices(new_dataset_dir, new_valid_indices)

    def get_img_file(self, i, load=True):
        # if load, then get filename for loading a file, otherwise assumed to be using for making a new file
        if load:
            assert i < self.total_ts_including_last_obs
        folder, ind = self._get_img_folder_index(i)
        fol_str = self.img_dir + '/' + str(folder).zfill(self.img_fol_str_len)
        if not load:
            os.makedirs(fol_str, exist_ok=True)
        return fol_str + '/' + str(ind).zfill(self.img_str_len) + self.img_type

    def get_depth_file(self, i, load=True):
        # if load, then get filename for loading a file, otherwise assumed to be using for making a new file
        if load:
            assert i < self.total_ts_including_last_obs
        folder, ind = self._get_img_folder_index(i)
        fol_str = self.depth_dir + '/' + str(folder).zfill(self.img_fol_str_len)
        if not load:
            os.makedirs(fol_str, exist_ok=True)
        return fol_str + '/' + str(ind).zfill(self.img_str_len) + '.npy'

    def _get_img_folder_index(self, i):
        return i // self.imgs_per_folder, i % self.imgs_per_folder

    def load_to_ram_worker(self, index, normalize_img=False, add_depth_dim=False):
        """ load a single index """
        img = np.array(Image.open(self.get_img_file(index)))
        if normalize_img:
            img = (img * .003921569).astype('float32')  #  1 / 255
        depth = np.load(self.get_depth_file(index))
        if add_depth_dim:
            depth = np.expand_dims(depth, axis=-1).astype('float32')
        return img, depth

    def load_to_ram_multiind_worker(self, index_range, normalize_img=False, add_depth_dim=False):
        """ load multiple indices """
        img_depths = []
        for i in index_range:
            img, depth = self.load_to_ram_worker(i, normalize_img, add_depth_dim)
            img_depths.append([img, depth])
        return img_depths

    def load_to_ram(self, num_workers=4, normalize_img=False, add_depth_dim=False,
                    load_unloaded=True, selected_valid_indices=None):
        """ Load dataset to ram for faster training. Normalizing the image here can save
        training time later at the extra expense of storing floats instead of uint8s in memory.

        If load_unloaded, load any indices that are not yet loaded. NOTE: this assumes
        that the data was loaded contiguously from 0 to a prev value of total_ts_including_last_obs.
        Any other way of loading the ram will break this.

        selected_valid_indices should be a set of valid indices to load, and the function will only load
        indices not previously loaded (and will also ensure the final obs of each traj are loaded as well)."""
        if self.ram_data is None or load_unloaded:
            if selected_valid_indices is not None:
                # TODO create a mapping between the called indices at the actual used indices
                # i.e. so all of the wanted indices are loaded contiguously from 0 to len(selected_valid_indices)
                # and then create a dict that contains maps from selected indices to actual indices
                sorted = np.sort(selected_valid_indices)
                streaks = np.split(sorted, np.where(np.diff(sorted) != 1)[0]+1)
                with_last = [np.append(e, e[-1]+1) for e in streaks]
                wanted_indices = np.hstack(with_last)
                indices_to_load = np.setdiff1d(wanted_indices, self._loaded_ram_indices)
            else:
                if load_unloaded and self.ram_data is not None:
                    indices_to_load = range(len(self.ram_data['img']), self.total_ts_including_last_obs)
                else:
                    indices_to_load = range(self.total_ts_including_last_obs)

            if len(indices_to_load) < 5000:
                data = self.load_to_ram_multiind_worker(indices_to_load, normalize_img, add_depth_dim)
            else:
                with Pool(processes=num_workers) as pool:
                    # data = pool.map(ImgReplayBufferDisk.load_to_ram_worker, range(self.dataset.total_ts_including_last_obs))
                    # data = pool.map(self.load_to_ram_worker, range(self.total_ts_including_last_obs))
                    data = pool.starmap(self.load_to_ram_worker, zip(indices_to_load,
                                                                 repeat(normalize_img), repeat(add_depth_dim)))

            if load_unloaded and self.ram_data is not None:
                if len(data) > 0:
                    self.ram_data['img'] = np.concatenate((self.ram_data['img'], np.array([item[0] for item in data])))
                    self.ram_data['depth'] = np.concatenate((self.ram_data['depth'], np.array([item[1] for item in data])))
                    self.ram_data['state'] = self.data['state_data'].astype('float32')
            else:
                self.ram_data = dict(
                    img=np.array([item[0] for item in data]),
                    depth=np.array([item[1] for item in data]),
                    state=self.data['state_data'].astype('float32'))

            print('RAM used by dataset: %.1f MB' % ((self.ram_data['img'].nbytes + self.ram_data['depth'].nbytes +
                                                     self.ram_data['state'].nbytes) / 1e6))
        else:
            print('RAM data already loaded. Flush ram data before calling load to ram again, or use load_unloaded.')

    def flush_ram(self):
        self.ram_data = None
        self.gpu_data = None
        self._loaded_ram_indices = np.array([])

    def new_dataset_from_indices(self, new_dir, indices):
        """ Generate a new dataset obj from a given set of indices"""
        if self.ind is not None:
            state_dim = self.ind['a']
            act_dim = self.ind['r'] - self.ind['a']
        else:
            state_dim = None
            act_dim = None
        new_dataset = Dataset(new_dir, img_type=self.img_type, imgs_per_folder=self.imgs_per_folder,
                              state_dim=state_dim, act_dim=act_dim)
        i_sorted = np.sort(indices)
        streaks = np.split(i_sorted, np.where(np.diff(i_sorted) != 1)[0] + 1)
        with_last = [np.append(e, e[-1] + 1) for e in streaks]
        for i_s, s in enumerate(with_last):
            data_list = list(self.data['state_data'][s])
            img_list = []; depth_list = []
            for i in s:
                img_list.append(np.array(Image.open(self.get_img_file(i))))
                depth_list.append(np.load(self.get_depth_file(i)))
            new_dataset.append_traj_data_lists(data_list, img_list, depth_list, final_obs_included=True)
            print('Copying traj %d of %d to new dataset at %s' % (i_s + 1, len(with_last), new_dir))
        return new_dataset

    def get_data(self, inds, normalize_img=True, add_depth_dim=True):
        """ Get some data given some indices """
        if self.ram_data is None:
            img_depths = self.load_to_ram_multiind_worker(inds, normalize_img, add_depth_dim)
            return (
                np.array([item[0] for item in img_depths]),
                np.array([item[1] for item in img_depths]),
                self.data['state_data'].astype('float32')[np.array(inds)])
        else:
            return (
                self.ram_data['img'][inds],
                self.ram_data['depth'][inds],
                self.ram_data['state'][inds])

    def get_traj_indices_as_list(self, traj_inds=None):
        """ Get data indices correponding to continuous trajectories as a list of arrays. If traj_inds is
            None, return all indices. Otherwise, return the indices corresponding to the trajectory indices
            (e.g. if you only want the i-th trajectory)"""
        inds_list = []
        if traj_inds is None:
            traj_inds = range(len(self.data['traj_lens']))
        for i in traj_inds:
            start_i = sum(self.data['traj_lens'][:i])
            start_i_valid = self.data['valid_indices'][start_i]
            final_i = start_i_valid + self.data['traj_lens'][i]
            inds_list.append(list(range(start_i_valid, final_i)))
        return inds_list

