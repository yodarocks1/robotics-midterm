import random

import gymnasium as gym
from gymnasium.envs.toy_text import frozen_lake

class FrozenLakeEnvModified(frozen_lake.FrozenLakeEnv):
    def __init__(self, death_probability: float = 0.02, **kwargs):
        self.death_probability = death_probability
        super().__init__(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if random.random() < self.death_probability:
            reward = 0
            terminated = True
        return (observation, reward, terminated, truncated, info)
