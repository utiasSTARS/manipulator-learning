import h5py
import numpy as np
import os
import shutil


class HDF5Dataset:
    def __init__(self, data_dir, example_obs):
        if type(example_obs) == dict:
            self.dict_obs = True
            self.names = list(example_obs.keys())
            self.img_names = []
            for n in self.names:
                if len(example_obs[n].shape) > 1:  # treat as image
                    self.img_names.append(n)
                    self.names.remove(n)
        else:
            self.dict_obs = False

        # generate config
        self.data_dir = data_dir
        self.data_file = data_dir + '/obs.hdf5'
        if os.path.exists(self.data_file):
            with h5py.File(self.data_file, 'r') as f:
                self.num_trajs = f['config'].attrs['num_trajs']
                self.timesteps_per_traj = list(f['config'].attrs['timesteps_per_traj'])
                self.total_timesteps = f['config'].attrs['total_timesteps']
        else:
            self.num_trajs = 0
            self.timesteps_per_traj = []
            self.total_timesteps = 0
        self.num_traj_timesteps = 0

    def save_dataset_info_to_file(self):
        print("Saving dataset info to file.")
        with h5py.File(self.data_file, 'a') as f:
            config = f.require_group('config')
            config.attrs.create('num_trajs', data=self.num_trajs)
            config.attrs.create('total_timesteps', data=self.total_timesteps)
            config.attrs.create('timesteps_per_traj', data=np.array(self.timesteps_per_traj))

    def delete(self, index):
        if self.num_trajs > 0:
            if self.dict_obs:
                for i_n in self.img_names:
                    dir = self.data_dir + '/' + i_n + '/run_' + str(index).zfill(4)
                    shutil.rmtree(dir)
                with h5py.File(self.data_file, 'a') as f:
                    for n in self.names:
                        del f[n + '/run_' + str(self.num_trajs).zfill(4)]

            else:
                with h5py.File(self.data_file, 'a') as f:
                    del f['run_' + str(self.num_trajs).zfill(4)]

            self.num_trajs -= 1
            self.total_timesteps -= self.timesteps_per_traj.pop(index)
            self.save_dataset_info_to_file()
            print("Demo %d deleted" % index)
        else:
            print("No demos to delete.")

    def add_traj_to_dataset(self, traj_data):
        pass

    def get_full_dataset(self):
        pass

    def get_data_run_timestep(self, run, ts):
        pass

    def get_data_index(self, index):
        pass