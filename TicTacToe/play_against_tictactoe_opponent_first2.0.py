import time

from gym import spaces
from gym import Env
from gym.spaces import Discrete
import numpy as np
from stable_baselines3 import PPO


class Board:
    def __init__(self):
        self.history = []
        self.history_coords = []


    def update_board(self, position, whichPlayer):
        # WhichPlayer = True  -   Player One (o)
        # WhichPlayer = False -   Player Two (/\)
        if not position in self.history_coords:

            self.history.append((position, int(whichPlayer == True)))
            self.history_coords.append(position)

    def render(self):
        print("Render:")
        for y in range(3):
            for x in range(3):
                if ((x, y), 1) in self.history:
                    print(' x ', end='')
                elif ((x, y), 0) in self.history:
                    print(' o ', end='')
                else:
                    print(' - ', end='')
            print('\n')
        print('-----------------')


    def getObservation(self):
        obs = np.zeros((3, 3))
        for i in self.history:
            obs[i[0][1], i[0][0]] = i[1] + 1
        return obs

    def reset(self):
        self.history = []
        self.history_coords = []


board = Board()

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

        return False, 0


    def step(self, action):
        done = False
        reward = 0

        if not self.action_defs[action + 1] in board.history_coords:
            board.update_board(self.action_defs[action + 1], self.player)
            board.render()

            if len(board.history) != 9 and self.is_terminal()[0] is False:
                print("It's your turn")
                opponent = self.opponent.decision(self.action_defs)

                board.update_board(opponent, not self.player)
                board.render()

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

class TicTacToeOpponent:
    def __init__(self, player=False):
        self.player = player
        self.model = PPO.load('./train/going_last/1.0/best_model_overall')

    def decision(self, action_def):

        X = input("Enter X COORD: ")
        Y = input("Enter Y COORD: ")
        return (int(X), int(Y))

o = TicTacToeOpponent()

env = TicTacToeENV(o, player=True)

model = PPO.load('./train/best_model_overall')

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

    print('Episode:{} Score:{}'.format(episode, score))
    overall += score

print("mean: " + str(overall / episodes))
env.close()
