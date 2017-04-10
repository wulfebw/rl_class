"""
MDP from class:
7 states, 1-based
    - since potentially fixed horizon, the state must be 2d
2 actions, left = 0 and right = 1
fixed horizon MDP
    - reflected in state
    - horizon of 10 means that you take 10 actions (i.e., step called 10 times)
"""
import gym
import random
from gym import spaces

class ClassEnv(gym.Env):
    def __init__(self, min_x=1, max_x=7, discount=1, horizon=10):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        self.min_x = min_x
        self.max_x = max_x
        self.discount = discount
        self.horizon = horizon
        self._reset()

    def _step(self, a):
        assert self.action_space.contains(a)
        x, t = self.s
        assert x >= self.min_x and x <= self.max_x
        assert t is None or t > 0 

        if self.horizon is not None:
            t -= 1

        # reward
        if x == self.min_x:
            r = 1
        elif x == self.max_x:
            r = 10
        else:
            r = 0

        # state
        if a == 0:
            x = max(x - 1, self.min_x)
        else:
            x = min(x + 1, self.max_x)

        self.s = (x, t)
        done = t == 0
        return self.s, r, done, {}

    def _reset(self):
        # state starts in the middle
        self.s = ((self.max_x + self.min_x) / 2, self.horizon)
        return self.s