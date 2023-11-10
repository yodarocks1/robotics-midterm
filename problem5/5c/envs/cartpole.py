import gymnasium as gym
from gymnasium.envs.classic_control import cartpole

class CartPoleEnvModified(cartpole.CartPoleEnv):
    def __init__(self, render_mode=None, left_multiplier=1., right_multiplier=1.):
        self.left_multiplier = left_multiplier
        self.right_multiplier = right_multiplier
        super(CartPoleEnvModified, self).__init__(render_mode=render_mode)
        self.base_force_mag = self.force_mag

    def step(self, action):
        if action == 0:
            self.force_mag = self.base_force_mag * self.left_multiplier
        else:
            self.force_mag = self.base_force_mag * self.right_multiplier
        return super().step(action)

