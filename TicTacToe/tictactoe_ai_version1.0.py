import random

from gym import spaces
from gym import Env
from gym.spaces import Discrete
import time, os
import numpy as np
from stable_baselines3 import PPO
import cv2
from stable_baselines3.common.callbacks import BaseCallback

CELL_SIZE = 100
SCREEN_SIZE = 500



# 145, 370 (Top Left Square Coord)

class Board:
    def __init__(self):
        self.circle_radius = 30
        self.picture = np.zeros(shape=(SCREEN_SIZE, SCREEN_SIZE, 3))
        self.picture.fill(255)
        self.history = []
        self.history_coords = []

    def draw_object(self, coords_list):
        if len(coords_list) == 1:
            self.picture = cv2.circle(self.picture, coords_list[0], self.circle_radius, (0, 0, 0), -1)
        if len(coords_list) == 3:
            cv2.line(self.picture, coords_list[0], coords_list[1], (0, 0, 0), 1)
            cv2.line(self.picture, coords_list[1], coords_list[2], (0, 0, 0), 1)
            cv2.line(self.picture, coords_list[0], coords_list[2], (0, 0, 0), 1)

    def _build_display(self):
        for col in range(2):
            x1, y1 = (col + 1) * CELL_SIZE + 100, CELL_SIZE + 20
            x2, y2 = (col + 1) * CELL_SIZE + 100, \
                     (2) * CELL_SIZE + 220
            cv2.line(self.picture, (x1, y1), (x2, y2), (0, 0, 0), 1)

        for row in range(2):
            x1, y1 = CELL_SIZE, (row + 1) * CELL_SIZE + 120
            x2, y2 = (2) * CELL_SIZE + 200, \
                     (row + 1) * CELL_SIZE + 120
            cv2.line(self.picture, (x1, y1), (x2, y2), (0, 0, 0), 1)

    def update_board(self, position, whichPlayer):
        # WhichPlayer = True  -   Player One (o)
        # WhichPlayer = False -   Player Two (/\)
        if not position in self.history_coords:
            if whichPlayer:
                self.draw_object([(150 + position[0] * CELL_SIZE, 370 - position[1] * CELL_SIZE)])

            if not whichPlayer:
                self.draw_object([(110 + position[0] * CELL_SIZE, 330 - position[1] * CELL_SIZE),
                                  (150 + position[0] * CELL_SIZE, 410 - position[1] * CELL_SIZE),
                                  (190 + position[0] * CELL_SIZE, 330 - position[1] * CELL_SIZE)])
            self.history.append((position, int(whichPlayer == True)))
            self.history_coords.append(position)

    def render(self, mode='human'):
        if mode == 'human':
            cv2.imshow('a', self.picture)
            cv2.waitKey(10)

            time.sleep(1)

    def getObservation(self):
        obs = np.zeros((3, 3))
        for i in self.history:
            obs[i[0][1], i[0][0]] = i[1] + 1
        return obs

    def reset(self):
        self.history = []
        self.history_coords = []
        self.picture = np.zeros(shape=(SCREEN_SIZE, SCREEN_SIZE, 3))
        self.picture.fill(255)
        self._build_display()


board = Board()
board._build_display()


class TicTacToeENV(Env):
    def __init__(self, opponent, player=True):
        self.player = player
        self.opponent = opponent
        self.action_defs = {1: (0, 2), 2: (1, 2), 3: (2, 2), 4: (0, 1),
                            5: (1, 1), 6: (2, 1), 7: (0, 0), 8: (1, 0),
                            9: (2, 0)}

        self.observation_shape = (3, 3)
        self.observation_space = spaces.Box(low=0, high=2, shape=self.observation_shape)

        self.action_space = Discrete(9)

    def is_terminal(self):
        # 0 for tie
        # 1 for winning
        # -1 for losing
        # 2 for nothing
        if len(board.history) == 9:
            self.reset()
            return True, 1
        # Checking Horizontally
        for i in range(3):
            if np.all(board.getObservation()[i, :] == int(self.player == True) + 1):
                return True, 2
            if np.all(board.getObservation()[i, :] == int(self.player == False) + 1):
                return True, -5

        # Checking Vertically
        for i in range(3):
            if np.all(board.getObservation()[:, i] == int(self.player == True) + 1):
                return True, 2
            if np.all(board.getObservation()[:, i] == int(self.player == False) + 1):
                return True, -5

        # Checking Diagonally
        if board.getObservation()[0, 0] == int(self.player == True) + 1 and \
                board.getObservation()[1, 1] == int(self.player == True) + 1 and \
                board.getObservation()[2, 2] == int(self.player == True) + 1:
            return True, 2
        if board.getObservation()[0, 0] == int(self.player == False) + 1 and \
                board.getObservation()[1, 1] == int(self.player == False) + 1 and \
                board.getObservation()[2, 2] == int(self.player == False) + 1:
            return True, -5
        if board.getObservation()[0, 2] == int(self.player == True) + 1 and \
                board.getObservation()[1, 1] == int(self.player == True) + 1 and \
                board.getObservation()[2, 0] == int(self.player == True) + 1:
            return True, 2
        if board.getObservation()[0, 2] == int(self.player == False) + 1 and \
                board.getObservation()[1, 1] == int(self.player == False) + 1 and \
                board.getObservation()[2, 0] == int(self.player == False) + 1:
            return True, -5

        return False, 2

    def render(self, render_mode='human'):
        board.render(render_mode)

    def step(self, action):
        done = False
        reward = 0


        if not self.action_defs[action + 1] in board.history_coords:

            board.update_board(self.action_defs[action + 1], self.player)
            if len(board.history) != 9 and self.is_terminal()[0] is False:
                board.update_board(self.opponent.decision(self.action_defs), not self.player)

            if self.is_terminal()[0]:
                reward = self.is_terminal()[1]
                done = True
        else:
            reward = -1
            done = False

        return board.getObservation(), reward, done, {}

    def reset(self):
        board.reset()
        return board.getObservation()


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


callback = TrainAndLoggingCallback(check_freq=100000, save_path='./train/going_last/1.0/logs')


class TicTacToeOpponent:
    def __init__(self, player=False):
        self.player = player

    def decision(self, action_def):
        items = []
        for key, value in action_def.items():
            if value not in board.history_coords:
                items.append(value)
        a = random.choice(items)
        return a



o = TicTacToeOpponent()

env = TicTacToeENV(o, player=True)

#model = PPO('MlpPolicy', env, tensorboard_log='./train/log', verbose=1, learning_rate=0.000005)
model = PPO.load('./train/best_model_overall')
#model.learn(total_timesteps=10000000, callback=callback)

episodes = 5
overall = 0
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action = model.predict(obs)[0]

        obs, reward, done, info = env.step(action)
        score += reward

        env.render()
    print('Episode:{} Score:{}'.format(episode, score))
    overall += score

print("mean: " + str(overall / episodes))
env.close()
