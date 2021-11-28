#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from torch.utils.data import Dataset
import os
import scipy.io as sio
#from skimage.transform import resize
#from skimage.transform import rotate
import numpy as np
from scipy import optimize

np.set_printoptions(threshold=np.inf, suppress=True)
import random
import copy
import math


class OffroadLoader(Dataset):
    def __init__(self, grid_size, train=True, demo=None, datadir='/home/ganlu/minicheetah_irldata/', pre_train=False, tangent=False,
                 more_kinematic=None):
        assert grid_size % 2 == 0, "grid size must be even number"
        self.grid_size = grid_size
        if train:
            self.data_dir = datadir + 'train_data/'
        else:
            self.data_dir = datadir + 'test_data_101_200/'

        if demo is not None:
            self.data_dir = datadir + '/irl_data/' + demo

        items = os.listdir(self.data_dir)
        self.data_list = []
        for item in items:
            self.data_list.append(self.data_dir + '/' + item)

        #self.data_normalization = sio.loadmat(datadir + '/irl_data/train-data-mean-std.mat')
        self.pre_train = pre_train

        # kinematic related feature
        self.center_idx = self.grid_size / 2
        self.delta_x_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.float)
        self.delta_y_layer = self.delta_x_layer.copy()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.delta_x_layer[x, y] = x - self.center_idx
                self.delta_y_layer[x, y] = y - self.center_idx

    def __getitem__(self, index):
        data_mat = sio.loadmat(self.data_list[index])
        feat, robot_state_feat, past_traj, future_traj, ave_energy_cons = data_mat['feat'].copy(), data_mat['robot_state_data'], data_mat['past_traj'], data_mat['future_traj'], data_mat['average_energy_consumption']
        normalization = 0.5 * self.grid_size
        feat = np.vstack((feat, np.expand_dims(self.delta_x_layer.copy() / normalization, axis=0)))
        feat = np.vstack((feat, np.expand_dims(self.delta_y_layer.copy() / normalization, axis=0)))
        
        # visualize rgb
        feat = np.vstack((feat, np.expand_dims(feat[2], axis=0)))
        feat = np.vstack((feat, np.expand_dims(feat[3], axis=0)))
        feat = np.vstack((feat, np.expand_dims(feat[4], axis=0)))
        # normalize features locally
        for i in range(5):
            feat[i] = (feat[i] - np.mean(feat[i])) / np.std(feat[i])
        # normalize robot state feature locally
        robot_state_feat = (robot_state_feat - np.mean(robot_state_feat, axis=0, keepdims=True)) / np.std(robot_state_feat, axis=0, keepdims=True)

 
        if self.pre_train:
            target = data_mat['feat'][1].copy()  # copy the variance layer first
            target[target < 0.5] = 0.0
            target[target >= 0.5] = -1.0
            return feat, target
       
        future_traj = self.auto_pad_future(future_traj[:, :2])
        past_traj = self.auto_pad_past(past_traj[:, :2])

        return feat, past_traj, future_traj, robot_state_feat, ave_energy_cons

    def __len__(self):
        return len(self.data_list)

    def auto_pad_past(self, traj):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        fixed_len = self.grid_size
        if traj.shape[0] >= self.grid_size:
            traj = traj[traj.shape[0]-self.grid_size:, :]
            #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
        pad_len = self.grid_size - traj.shape[0]
        pad_array = np.full((pad_len, 2), np.nan)
        output = np.vstack((traj, pad_array))
        return output

    def auto_pad_future(self, traj):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        fixed_len = self.grid_size
        if traj.shape[0] >= self.grid_size:
            traj = traj[:self.grid_size, :]
            #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
        pad_len = self.grid_size - traj.shape[0]
        pad_array = np.full((pad_len, 2), np.nan)
        output = np.vstack((traj, pad_array))
        return output
