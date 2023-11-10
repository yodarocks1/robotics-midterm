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
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    def forward(self, x):
        # Ensure that x has the correct shape (batch_size, n_observations)
        x = x.view(x.size(0), -1)
        return self.layers(x)


class EnvModel:
    DEFAULT_VALUES = {
        "batch_size": 128,
        "γ": 0.99,
        "ε_start": 0.9,
        "ε_end": 0.05,
        "ε_decay": 10000,
        "τ": 0.005,
        "lr": 1e-4,
    }
    def __init__(self, env, **kwargs):
        self.env = env
        self.values = dict(**EnvModel.DEFAULT_VALUES)
        self.values.update(kwargs)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state, info = self.env.reset()
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
                return self.policy_net(state).argmax().view(1, 1)
        else:
            action = self.env.action_space.sample()
            return torch.tensor([[action]], device=self.device, dtype=torch.long)
    def _plot_durations(self, show_result=False):
        fig = plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        fig.clear()
        if show_result:
            plt.title('Result')
        else:
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
    def train(self, num_episodes: int = 100):
        plt.ion()
        self.training_completion = 0
        for episode in range(num_episodes):
            self.training_completion = episode / num_episodes

            # Initialize the environment and state
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            total_reward = 0
            for t in count():
                # Select and perform an action
                action = self._select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                total_reward += reward

                # Store the transition in memory
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated  # Consider both termination and truncation

                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Optimize the model
                self._optimize_model()

                if done:
                    # Update the target network every few episodes
                    if episode % self.values["τ"] == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                    # Record the episode duration and reward
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(total_reward)

                    # Plot the training progress
                    self._plot_durations()

                    break

        self.training_completion = 1
        print('Training complete')
        self._plot_durations(show_result=True)
        plt.ioff()
        plt.show()

    def _optimize_model(self):
        if len(self.memory) < self.values["batch_size"]:
            return

        transitions = self.memory.sample(self.values["batch_size"])
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute max Q'(s', a') for all next states.
        next_state_values = torch.zeros(self.values["batch_size"], device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.values["γ"]) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    def save(self, path):
        torch.save(self.target_net.state_dict(), path)


