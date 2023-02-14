from gym import spaces
from gym import Env
import mss
from gym.spaces import Discrete
import pytesseract as pt
import pyautogui
import numpy as np
import time
from stable_baselines3 import PPO
import cv2, os
from stable_baselines3.common.callbacks import BaseCallback

pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
monitor = {"top": 393, "left": 236, "width": 1148, "height": 393}

class DinoAgent(Env):
    def __init__(self):
        self.cap = mss.mss()
        self.observation_space = spaces.Box(low=0, high=255, shape=(393, 1148))
        self.action_space = Discrete(3)

    def get_observation(self):
        img = self.cap.grab(monitor)
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def get_done(self):
        return "GAME OVER" in pt.image_to_string(self.get_observation())

    def step(self, action):
        reward = 0
        done = self.get_done()
        if action == 1:
            pyautogui.press('space')
        if action == 2:
            pyautogui.press('down')
        if done is False:
            reward = 0.05
        if done is True:
            reward = -2
        return self.get_observation(), reward, done, {}

    def reset(self):
        time.sleep(1)
        pyautogui.click(x=1920/2, y=1080/2)
        return self.get_observation()
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True


callback = TrainAndLoggingCallback(check_freq=100, save_path='./train/logs')


env = DinoAgent()
episodes = 5
overall = 0

model = PPO('MlpPolicy', env, tensorboard_log='./train/log', verbose=1, learning_rate=0.000005)
model.learn(total_timesteps=1000, callback=callback)

for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward

    print('Episode:{} Score:{}'.format(episode, score))
    overall += score

print("mean: " + str(overall / episodes))
env.close()
