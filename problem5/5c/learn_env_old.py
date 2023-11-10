import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        print(x, end='')
        y = self.layers(x)
        print(" =>", y)
        return y


class EnvModel:
    DEFAULT_VALUES = {
        "batch_size": 128,
        "γ": 0.99,
        "ε_start": 0.9,
        "ε_end": 0.05,
        "ε_decay": 1000,
        "τ": 0.005,
        "lr": 1e-4,
    }
    def __init__(self, env, **kwargs):
        self.env = env
        self.values = dict(**EnvModel.DEFAULT_VALUES)
        self.values.update(kwargs)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state, info = self.env.reset()
        self.state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.n_actions = self.env.action_space.n
        if hasattr(state, "__len__"):
            self.n_observations = len(state)
        else:
            self.n_observations = 1

        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.values["lr"], amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.episode_durations = []
        self.episode_rewards = []
        self.steps_done = 0
    def _select_action(self, state):
        sample = random.random()
        ε_threshold = self.values["ε_end"] + (self.values["ε_start"] - self.values["ε_end"]) * \
            math.exp(-1. * self.steps_done / self.values["ε_decay"])
        self.steps_done += 1
        if sample > ε_threshold:
            with torch.no_grad():
                return self.policy_net(self.state).argmax().view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
    def _plot_durations(self, show_result=False):
        fig = plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            fig.clear()
            plt.title('Training...')
        if len(fig.axes) == 0:
            fig.add_axes(matplotlib.axes.Axes(fig))
        if len(fig.axes) == 1:
            fig.add_axes(fig.axes[0].twinx())
        ax2, ax1 = fig.axes
        ax1.set_xlabel('Episode')

        color = 'tab:red'
        ax1.set_ylabel('Duration', color=color)
        ax1.plot(durations_t.numpy(), color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        bottom, top = ax1.get_ylim()
        if bottom > 0 and top > 0:
            ax1.set_ylim(bottom=0)
        elif bottom < 0 and top < 0:
            ax1.set_ylim(top=0)
        
        color = 'tab:blue'
        ax2.set_ylabel('Reward', color=color)
        ax2.plot(rewards_t.numpy(), color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        bottom, top = ax2.get_ylim()
        if bottom > 0 and top > 0:
            ax2.set_ylim(bottom=0)
        elif bottom < 0 and top < 0:
            ax2.set_ylim(top=0)
        
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax1.plot(means.numpy(), color='tab:orange')
        fig.tight_layout()
        plt.pause(0.001)
    def _optimize(self):
        if len(self.memory) < self.values['batch_size']: return
        transitions = self.memory.sample(self.values['batch_size'])

        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        print("State Batch:", state_batch)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.values['batch_size'], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.values['γ']) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    def train(self, num_episodes=None):
        plt.ion()
        if num_episodes is None:
            if torch.cuda.is_available():
                num_episodes = 100
            else:
                num_episodes = 50
        for i_episode in range(num_episodes):
            episode_rewards = 0
            state, info = self.env.reset()
            self.state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self._select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_rewards += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(self.state, action, next_state, reward)
                self.state = next_state
                self._optimize()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.values['τ'] + target_net_state_dict[key]*(1-self.values['τ'])
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(episode_rewards)
                    self._plot_durations()
                    break
            print("E" + str(i_episode))
        self.env.close()
        print('Complete')
        self._plot_durations(show_result=True)
        plt.ioff()
        plt.show()
    def save(self, path):
        torch.save(self.target_net.state_dict(), path)


