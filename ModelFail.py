import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class ModelFail(gym.Env):
    """ModelFail Enviroment:

    See [https://arxiv.org/pdf/1604.00923.pdf] for a description of the
    original environment. The following description is directly based off the
    descriptoin in the linked paper.

    Although the MDP has three states, the agent does not observe which state
    it is in. It always starts in the leftmost state.

    It has two actions available. The first action moves it to the upper state
    whereas the second action moves it to the lower state. If action was in the
    upper state reward is r, while reward is -r if the agent was in the lower
    state.

    When the agent reaches these two states, the agent's two possible actions
    always moves it to the end state.

    Horizon H = 2.

    Use the following numbering system:
    Left state: 0
    Upper state: 1
    Lower state: 2
    Terminal state: 3
    """
    def __init__(self, r=1):
        self.state = 0
        self.r = r
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if self.state == 1:
            # Moves agent to the terminal state, denoted by 3
            self.state = 3
            reward = self.r
        elif self.state == 2:
            # Moves agent to the terminal state, denoted by 3
            self.state = 3
            reward = -self.r
        elif action == 0:
            # Agent is at left state and moves up, no reward recieved
            self.state = 1
            reward = 0
        else:
            # Agent is at left state and moves down, no reward recieved
            self.state = 2
            reward = 0
        done = (self.state == 3)
        return 0, reward, done, {} #observed is always 0

    def reset(self):
        self.state = 0
        return 0 #observed is always 0
