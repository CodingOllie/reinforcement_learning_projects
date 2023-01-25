from gym import spaces
from gym import Env
import numpy as np
import random, pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

SCREEN_SIZE = 500
OBSTICLE_SIZE = [(40, 40), (50, 40), (60, 40), (20, 50)]

class DinoENV(Env):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode([SCREEN_SIZE, SCREEN_SIZE])

        self.score = 0
        self.addToScore = 1
        self.obsticles = []

        self.gravity = 9.8
        self.weight = 1
        self.velY = 0
        self.y = 0

        self.create_obsticles()

        self.observation_space = spaces.Dict({
            'player position':
            spaces.Box(low=0, high=300),
            'obsticles':
            self.obsticles
        })
        self.action_space = spaces.Discrete(2)


    def create_obsticles(self):
        if self.score == 0:
            a = random.choice(OBSTICLE_SIZE)
            b = random.choice(OBSTICLE_SIZE)
            self.obsticles.append((600, 0, a[0], a[1]))
            self.obsticles.append((1000, 0, b[0], b[1]))
            self.score += self.addToScore
        else:
            for i in self.obsticles:
                if i[0] <= -10:
                    self.obsticles.pop(self.obsticles.index(i))

                    a = random.choice(OBSTICLE_SIZE)
                    self.obsticles.append((800, 0, a[0], a[1]))
    def apply_gravity(self):
        if self.y > 0:
            self.velY -= (self.gravity * self.weight) / self.velY

            self.y += self.velY
        if self.y < 0:
            self.y = 0

    def is_terminal(self):
        for i in self.obsticles:
            if 10 < i[0] < 50:
                if self.y == 0:
                    return True
        return False

    def step(self, action):
        done = False
        reward = 0
        if action == 1:
            self.velY = 4
            self.y = 0.1

        self.apply_gravity()
        self.create_obsticles()

        if self.is_terminal():
            done = True
            reward = -5
        else:
            done = False
            reward = 0.5

        for i in self.obsticles:
            i[0] -= self.addToScore

        return {'player_position': self.y, 'obsticles': self.obsticles}, reward, done, {}
    def render(self, render_mode='human'):
        if render_mode == 'human':
            self.screen.fill((255, 255, 255))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.draw.rect(self.screen, (0, 0, 0), (10, self.y+70, 70, 100))

            pygame.display.flip()



    def reset(self):
        self.score = 0
        self.addToScore = 1
        self.obsticles = []

        self.create_obsticles()
        self.gravity = 9.8
        self.weight = 1
        self.velY = 0
        self.y = 0
        return {'player_position': self.y, 'obsticles': self.obsticles}

env = DinoENV()

#model = PPO.load('./train/best_model_overall')
model = PPO('MlpPolicy', env, tensorboard_log='./train/log', verbose=1, learning_rate=0.000005)
while True:
    env.render()


"""
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
"""



