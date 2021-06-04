#import the libraries
import numpy as np
import os
import time
import sys
import gym
from gym import error, spaces, utils
import copy
from gym.utils import seeding
import matplotlib.pyplot as plt

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STAY = 4

class GridEnv(gym.Env):
    
    num_env = 0
    def __init__(self, n=10): #'random'):
        self.n = n        
        self.n_states = self.n ** 2 
        self.state_reward =  np.full(100, -1)
        self.terminal_state = self.coord2ind([0,0])
        self.state_reward[self.terminal_state]=100
        self.absorbing_state = self.terminal_state
        self.done = False
        self.start_state = np.random.rand(n**2-1) #if not isinstance(start_state, str) else np.random.rand(n**2)
        self._reset()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(self.n_states) # with absorbing state
        #self._seed()
        #seeding

    def ind2coord(self, index):
        assert(index >= 0)
        #assert(index < self.n_states - 1)
        col = index // self.n
        row = index % self.n
        return [row, col]

    def coord2ind(self, coord):
        [row, col] = coord
        assert(row < self.n)
        assert(col < self.n)
        return col * self.n + row

    def is_terminal_state(self,state):
       #if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
        if self.state_reward[state] == -1:
          self.done = False
          return False
        else:
          self.done = True
          return True
          
    def _reset(self):
        self.state = np.random.randint(self.n_states-1)
        while self.is_terminal_state(self.state):
          self._reset()
        return self.state
        
    def seed(self, seed=None):
            self.np_random, seed = seeding.np.random(seed)
            return [seed]
        
    def _get_reward(self,new_state=None):
        reward = self.state_reward[new_state]
        return reward  

    def step(self, action):
        assert self.action_space.contains(action)
        if self.is_terminal_state(self.state):
            self.state = self.absorbing_state
            self.done = True
            return self.state, self._get_reward(), self.done, None
        [row, col] = self.ind2coord(self.state)
        #  if actions[action_index] == 'up' and current_row_index > 0:
        if action == UP:
            row = max(row - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.n - 1)
        elif action == RIGHT:
            col = min(col + 1, self.n - 1)
        elif action == LEFT:
            col = max(col - 1, 0)
        elif action == STAY:
            row = row
            col = col
        new_state = self.coord2ind([row, col])
        reward = self._get_reward(new_state=new_state)
        self.state = new_state
        action = self.actions
        return self.state, reward, self.done, None
