
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.nn import *
import numpy as np
import sys
import os
import gym
from gym import spaces
import glob
import sys
import keras as keras
import six
from six.moves import cPickle as pickle
import scipy as scipy
import scipy.io as io
from tqdm import tqdm

lib_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(lib_path)
import dataset

from keras.models import load_model
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer

import matplotlib as mpl
import matplotlib.image as mpimg
import random
from matplotlib import pyplot as plt

class NorbEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    max_index = 972
    min_index = 0
    num_actions = 4
    my_path = os.path.abspath(os.path.dirname(__file__))
    dataset_loc = os.path.join(my_path, 'dataset_norb.p')
    max_ep_time = 1500
    scenarios = ['dense_reward', 'sparse_reward', 'one_shot']
    elevation_angles = 9
    azimuth_angles = 18
    max_traj_len = elevation_angles + azimuth_angles - 2
                    

    """
    Description:
        A toy is turned around (azimuth / elevation changes) until it matches a
        target viewpoint
    Source:
        Our [TNS group, INI, RUB] own creation --- based on the NORB data set
    Observation:
        Type: Numpy Array with shape (96, 96, 4)
        Slice	Data description
        Obs[:, :, 0]	Left camera view on current object
        Obs[:, :, 1]	Right camera view on current object
        Obs[:, :, 2]	Left camera view on target object
        Obs[:, :, 3]	Right camera view on target object

    Actions:
        Type: Discrete(4)
        Num	Action
        0	 Rotate object 20° on turn table
        1	 Rotate object -20° on turn table
        2	 Elevate camera by 5°
        3	 Lower camera by 5°

    Reward:
        Reward is 1 if the viewpoint is the same as the target viewpoint,
        otherwise -1
    Starting State:
        The toy is assigned a random viewpoint and the target viewpoint is found
    Episode Termination:
        (TODO: Add "Episode length is greater than 200" if it's clever)
        Solved Requirements
        Considered solved when the current viewpoint matches the target
        viewpoint
    """
    # template from https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai

    def __init__(self, scenario=None):
        super(NorbEnv, self).__init__()
        if scenario is None:
            scenario = self.scenarios[0]

        if scenario == self.scenarios[2]:
            self.action_space = spaces.MultiDiscrete([self.max_traj_len, self.num_actions])
        else:
            self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(0, 255, (96, 96, 4), dtype=np.int)
        self.item = self.set_item(14, 'train')
        self._seed = 42
        self.scenario = scenario
        self.ep_time_left = None
        self.current_index = None
        self.target_index = None
        self.reset()

    def seed(self, seed=None):
        self._seed = np.random.randint(np.iinfo(np.int32).max, dtype=np.int32) if seed is None else seed

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.scenario == self.scenarios[2]:
            for a in action:
                self._take_action(a)
            self.ep_time_left == 1  # set episode time to zero effectively
        else:
            self._take_action(action)
        reward = self._get_reward()
        ob = np.stack((self.item[self.current_index].image_lt,
                            self.item[self.current_index].image_rt,
                            self.item[self.target_index].image_lt,
                            self.item[self.target_index].image_rt), axis=2) #ob = self.env.getState()
        self.ep_time_left -= 1
        if self.ep_time_left == 0:
            episode_over = True
        else:
            over, _ = self._is_close()
            episode_over = over


        return ob, reward, episode_over, {}

    def reset(self):
        self.current_index = np.random.randint(self.max_index)
        self.target_index = np.random.randint(self.max_index)
        ob = np.stack((self.item[self.current_index].image_lt,
                            self.item[self.current_index].image_rt,
                            self.item[self.target_index].image_lt,
                            self.item[self.target_index].image_rt), axis=2)
        ##ob = ob[np.newaxis, np.newaxis, :]
        self._seed = np.random.rand()
        self.ep_time_left = self.max_ep_time
        return ob

    def render(self, mode='human', close=False):
        if mode == 'human':
            plt.imshow(np.concatenate([self.item[self.current_index].image_lt, self.item[self.target_index].image_lt], axis=1))
            plt.ion()
            plt.pause(0.00001)
        if mode == 'rgb_array':
            return self.item[self.current_index].image_lt

    def _take_action(self, action):
        item = self.item
        b4_idx = self.current_index
        # actions mean in camera hemisphere space: left, right, up, down
        if action == 0:
            after_index = [i for i in range(len(item)) if item[i].azimuth==(item[b4_idx].azimuth+2)%36 and item[i].lighting==item[b4_idx].lighting and item[i].elevation==item[b4_idx].elevation ]
        if action == 1:
            after_index = [i for i in range(len(item)) if item[i].azimuth==(item[b4_idx].azimuth-2)%36 and item[i].lighting==item[b4_idx].lighting and item[i].elevation==item[b4_idx].elevation ]
        if action == 2:
            after_index = [i for i in range(len(item)) if item[i].azimuth==item[b4_idx].azimuth and item[i].lighting==item[b4_idx].lighting and item[i].elevation==(min(8, item[b4_idx].elevation+1)) ]
        if action == 3:
            after_index = [i for i in range(len(item)) if item[i].azimuth==item[b4_idx].azimuth and item[i].lighting==item[b4_idx].lighting and item[i].elevation==(max(0, item[b4_idx].elevation-1)) ]
        try: self.current_index = after_index[0]
        except: print(after_index)

    def _viewpoint_dist(self, vp1, vp2):
        # ignore lighting conditions
        return abs(vp1.elevation - vp2.elevation) + min(36 - abs(vp1.azimuth - vp2.azimuth), abs(vp1.azimuth - vp2.azimuth))

    def _is_close(self):
        i = self.current_index
        t = self.target_index
        dist = self._viewpoint_dist(self.item[i], self.item[t])
        return np.isclose(dist, 0), dist

    def _get_reward(self):
        close, dist = self._is_close()
        reward = 0

        if close:
            reward = 2
        else:
            if self.scenario == self.scenarios[0]:  # dense rewards
                reward = -1 * dist/50.
            else:  # sparse rewards or one-shot
                reward = 0

        return reward


    def set_item(self, seq_no, instance):
        try:
            dataset = pickle.load(open(self.dataset_loc, 'rb'))
            lists = dataset.group_dataset_by_category_and_instance(instance)
            del dataset
            return lists[seq_no]
        except:
            print('Please provide norb dataset at location \"{}\"'.format(self.dataset_loc))
            quit(-1)

