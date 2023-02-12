from gym import spaces
from gym import Env
from gym.spaces import Discrete
import time, os
import numpy as np
from stable_baselines3 import PPO
import cv2
from stable_baselines3.common.callbacks import BaseCallback

SCREEN_SIZE = 500

class Pong(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_SIZE, SCREEN_SIZE, 3))
        self.screen = np.zeros(shape=(SCREEN_SIZE, SCREEN_SIZE, 3))
        self.velXBALL, self.velYBALL = 0, 0
        self.xBAll, self.YBALL = 0, 0
        self.y = SCREEN_SIZE/2
        self.height =  SCREEN_SIZE//8

    def collide(self):
        if 10 < self.xBAll < 20:
            if self.velYBALL > self.y - self.height/2:

    def step(self, action):
        action -= 1
        self.y += action
