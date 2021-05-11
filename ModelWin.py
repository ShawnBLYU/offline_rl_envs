import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class ModelWin(gym.Env):
    """ModelWin Enviroment:

    See [https://arxiv.org/pdf/1604.00923.pdf] for a description of the 
    original environment. The following description is directly based off the
    descriptoin in the linked paper.
    
    Agent always starts in s1 and has two possible actions. 

    When agent is at s1, action 1 transitions to s2 with probability p, to
    s3 with probability 1 - p. Action 2 transitions to s2 with probability 1 -
    p and to s3 with probability p. If the agent transitions to s2, it receives
    a reward of r, and it if transitions to s3 it receives a reward of -r.

    In states s2 and s3, the agent has two actions but both always returns the
    agent to s1 and produce a reward of 0.

    Horizon H = 20.
    """
    def __init__(self, r=1, p=0.4):
        self.state = 0
        self.r = r
        self.p = p
        self.h = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if self.state != 0:
            # Agent is not at s1. Transitions the agent back to s1, receives
            # a reward of 0
            self.state = 0
            reward = 0
        elif action == 0:
            # Agent is at s1 and chooses action 1
            if self.np_random.rand() < self.p:
                # Agent transitions to s2 with probability p
                self.state = 1
                reward = self.r
            else:
                # Agent transitions to s3 with probability 1 - p
                self.state = 2
                reward = -self.r
        else:
            # Agent is at s1 and chooses action 2
            if self.np_random.rand() >= self.p:
                # Agent transitions to s2 with probability 1 - p
                self.state = 1
                reward = self.r
            else:
                # Agent transitions to s3 with probability p
                self.state = 2
                reward = -self.r
        self.h = self.h + 1
        done = (self.h == 20) # If the agent has stepped 20 times already

        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        self.h = 0
        return self.state
