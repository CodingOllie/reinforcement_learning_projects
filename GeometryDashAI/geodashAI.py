from gym import spaces
from gym import Env
from gym.spaces import Discrete
import numpy as np
from stable_baselines3 import PPO
import pygame, math
from stable_baselines3.common.callbacks import BaseCallback


def quadratic_formula(a, b, c):
    a1 = ((-1 * b) + math.sqrt((b ** 2 - (4 * a * c)))) / (2 * a)
    b1 = ((-1 * b) + -1 * (math.sqrt((b ** 2 - (4 * a * c))))) / (2 * a)

    return (a1, b1)


class GeoDashAI(Env):
    def __init__(self, next_spike=3):
        self.a = -3
        self.b = 0
        self.c = 2
        self.window = None
        self.render_true = False
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=500, shape=(2, 1)),
            'spike_position': spaces.Box(low=0, high=500, shape=(next_spike * 2, 1)),
            'platform': spaces.Box(low=0, high=500, shape=(next_spike * 4, 1))
        })
        self.next_spike = next_spike
        self.y = 0
        self.x = 0
        self.jump_x = min(quadratic_formula(self.a, self.b, self.c))
        self.jump = False
        self.spike_list = self.getLevel(0)
        self.platform_list = self.getLevel(1)
        self.action_space = Discrete(2)

    def getObservation(self):
        next_spikes = []
        next_platforms = []
        if len(self.getLevel(0)) >= self.next_spike:
            for i in range(self.next_spike):
                next_spikes.append(self.getLevel(0)[i])
        if len(self.getLevel(0)) < self.next_spike:
            for i in range(len(self.getLevel(0))):
                next_spikes.append(self.getLevel(0)[i])

        if len(self.getLevel(1)) >= self.next_spike:
            for i in range(self.next_spike):
                next_platforms.append(self.getLevel(1)[i])

        if len(self.getLevel(1)) < self.next_spike:
            for i in range(len(self.getLevel(1))):
                next_spikes.append(self.getLevel(1)[i])

        return {'position': (self.x, self.y), 'spike_position': tuple(next_spikes),
                'platform': tuple(next_platforms)}

    def getLevel(self, which):
        la = []
        with open('level') as l:
            lines = l.readlines()
            for i in lines:
                i = eval(i)
                if len(i) == 2 and which == 0:
                    la.append(i)
                if len(i) == 4 and which == 1:
                    la.append(i)
        return la

    def gravity(self):
        # -3x^2 + 2
        if not self.jump_x >= max(quadratic_formula(self.a, self.b, self.c)) or self.jump is True:
            if self.getCollisionPLAT()[0] is False:
                y = (self.a * (self.jump_x) ** 2 + self.b * self.jump_x + self.c)*60

                self.jump_x += 0.016
        if self.jump_x >= max(quadratic_formula(self.a, self.b, self.c)) or self.jump is False:
            if self.getCollisionPLAT()[0] is False:
                self.jump_x = min(quadratic_formula(self.a, self.b, self.c))
                self.jump = False
                y = 0
        platform = self.getCollisionPLAT()[1]
        if self.getCollisionPLAT()[0]:
            self.jump = False
            y = -platform[1] + platform[3]
        return round(y, 3)

    def step(self, action):
        reward = 0
        self.y = self.gravity()
        if action == 1:
            self.jump = True
        self.x += 2
        reward += 0.0002
        done = self.getCollisionSPIKE()
        if done == False:
            done = self.getCollisionPLAT()[2]

        return self.getObservation(), reward, done, {}

    def getCollisionSPIKE(self):
        collide = []
        for i in self.spike_list:
            if i[0] - self.x <= 10 and i[0] - self.x + 30 >= 10:
                if i[1] + 450 >= -self.y + 450 and i[1] + 400 <= -self.y + 450:
                    collide.append(True)
                else:
                    collide.append(False)
            else:
                collide.append(False)
        return True in collide

    def getCollisionPLAT(self):
        collide = []
        platform = None
        dead = False
        for i in self.platform_list:
            if i[0] + i[2] >= self.x >= i[0]:
                if -i[1] <= self.y <= -i[1] + i[3] and self.jump_x >= 0:
                    collide.append(True)
                    platform = i
                collide.append(False)
            collide.append(False)
        return True in collide, platform, dead

    def render(self, render_mode='human'):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((500, 500))
        self.window.fill((255, 255, 255))

        pygame.draw.rect(self.window, (0, 0, 0), (10, -self.y + 450, 50, 50))
        for i in self.spike_list:
            pygame.draw.rect(self.window, (255, 0, 0), (i[0] - self.x, i[1] + 450, 30, 50))
        for i in self.platform_list:
            pygame.draw.rect(self.window, (0, 255, 0), (i[0] - self.x, i[1] // 10 + 450, i[2], i[3]))

        pygame.display.flip()

    def reset(self):
        self.y = 0
        self.x = 0
        self.jump_x = min(quadratic_formula(self.a, self.b, self.c))
        self.jump = False
        self.spike_list = self.getLevel(0)
        self.platform_list = self.getLevel(1)

        return self.getObservation()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


env = GeoDashAI()
env.render_true = True
episodes = 1
overall = 0
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        score += reward

    print('Episode:{} Score:{}'.format(episode, score))
    overall += score

print("mean: " + str(overall / episodes))
env.close()
