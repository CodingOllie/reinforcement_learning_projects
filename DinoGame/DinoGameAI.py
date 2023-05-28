from gym import spaces
from gym import Env
import mss
from gym.spaces import Discrete
import pytesseract as pt
import pyautogui
import numpy as np
import time
from stable_baselines3 import DQN
import cv2, os
from stable_baselines3.common.callbacks import BaseCallback

pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
monitor = {"top": 393, "left": 236, "width": 1148, "height": 393}

class DinoAgent(Env):
    def __init__(self):
        self.cap = mss.mss()
        self.observation_space = spaces.Box(low=0, high=255, shape=(393, 1148))
        self.action_space = Discrete(3)
        self.high_score = 0
        self.startTime = time.time()

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
            reward = 0.3
            pyautogui.press('space')
        if action == 2:
            reward = 0.3
            pyautogui.press('down')
        if done is False and time.time() - self.startTime > 2:
            reward += 1
        if done is True:
            reward += -6
        return self.get_observation(), reward, done, {}

    def reset(self):
        time.sleep(1)
        pyautogui.click(x=1920/2, y=1080/2)

        self.startTime = time.time()
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


callback = TrainAndLoggingCallback(check_freq=1000, save_path='./train/logs')


env = DinoAgent()
episodes = 5
overall = 0

#model = PPO('MlpPolicy', env, tensorboard_log='./train/log', verbose=1, learning_rate=0.000005)
#model = PPO.load(env=env, path='train/logs/best_model_2000')
#model.learn(total_timesteps=700000, callback=callback)
model = DQN('MlpPolicy', env, tensorboard_log='./train/log', verbose=1, buffer_size=5000)
#model = DQN.load(env=env, path='./train/logs/best_model_8000.zip')
model.learn(total_timesteps=200000, callback=callback)

for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward

    print('Episode:{} Score:{}'.format(episode, score))
    overall += score

print("mean: " + str(overall / episodes))
env.close()
