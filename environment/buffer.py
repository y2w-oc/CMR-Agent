import torch
import numpy as np
import functools

from config import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cat(list_of_tensors, dim=0):
    """
    Concatenate a list of tensors.
    """
    return functools.reduce(lambda x, y: torch.cat([x, y], dim=dim), list_of_tensors)


def catcat(list_of_lists_of_tensors, dim_outer=0, dim_inner=0):
    """
    Recursively concatenate a list of tensors.
    """
    return cat([cat(inner_list, dim_inner) for inner_list in list_of_lists_of_tensors], dim_outer)


def discounted(vals, gamma=0.99):
    """
    Computes the discounted sum as used for the return in RL.
    """
    G = 0
    discounted = torch.zeros_like(vals)
    for i in np.arange(vals.shape[-1]-1, -1, -1):
        G = vals[..., i] + gamma * G
        discounted[..., i] = G
    return discounted


def advantage(rewards, values, gamma=0.99, gae_lambda=0):
    """
    Computes the advantage of the given returns as compared to the estimated values, optionally using GAE.
    """
    if gae_lambda == 0:
        returns = discounted(rewards, gamma)
        advantage = returns - values
    else:
        # Generalized Advantage Estimation (GAE) https://arxiv.org/abs/1506.02438
        # via https://github.com/inoryy/reaver/blob/master/reaver/agents/base/actor_critic.py
        values = torch.cat([values, torch.zeros((values.shape[0], 1, 1)).to(DEVICE)], dim=2)
        deltas = rewards + gamma * values[..., 1:] - values[..., :-1]
        advantage = discounted(deltas, gamma * gae_lambda)

    return advantage


class Buffer:

    """
    Utility class to gather a replay buffer. Computes returns and advantages over logged trajectories.
    """

    def __init__(self, config):
        self.config = config

        self.count = 0  # number of trajectories
        # environment
        self.states_2d = []
        self.states_3d = []
        self.state_values = []
        self.rewards = []

        # expert related
        self.expert_actions_r = []
        self.expert_actions_t = []

        # student related
        self.actions_r = []
        self.actions_t = []
        self.actions_logprob = []

    def __len__(self):
        return self.count

    def start_trajectory(self):
        """
        Initializes the list into which all samples of a trajectory are gathered.
        """
        #
        self.count += 1
        self.states_2d += [[]]
        self.states_3d += [[]]
        self.state_values += [[]]
        self.rewards += [[]]

        self.expert_actions_r += [[]]
        self.expert_actions_t += [[]]

        self.actions_r += [[]]
        self.actions_t += [[]]
        self.actions_logprob += [[]]

    def log_step(self, state_2d, state_3d, state_value, reward, expert_action_r, expert_action_t,
                 action_r, action_t, action_logprob
                 ):
        """
        Logs a single step in a trajectory.
        """
        self.states_2d[-1].append(state_2d.detach())
        self.states_3d[-1].append(state_3d.detach())
        self.state_values[-1].append(state_value.detach())
        self.rewards[-1].append(reward.detach())

        self.expert_actions_r[-1].append(expert_action_r.detach())
        self.expert_actions_t[-1].append(expert_action_t.detach())

        self.actions_r[-1].append(action_r.detach())
        self.actions_t[-1].append(action_t.detach())
        self.actions_logprob[-1].append(action_logprob.detach())

    def get_returns_and_advantages(self):
        """
        Computes the return and advantage per trajectory in the buffer.
        """
        # for rewards in self.rewards:
        #     print(cat(rewards, dim=-1).shape)
        #
        # for rewards, values in zip(self.rewards, self.state_values):
        #     print(cat(rewards, dim=-1).shape, cat(values, dim=-1).shape)

        returns = [discounted(cat(rewards, dim=-1), self.config.GAMMA).transpose(2, 1)
                   for rewards in self.rewards]  # per trajectory
        advantages = [advantage(cat(rewards, dim=-1), cat(values, dim=-1), self.config.GAMMA, self.config.GAE_LAMBDA).transpose(2, 1)
                      for rewards, values in zip(self.rewards, self.state_values)]
        return returns, advantages

    def get_samples(self):
        """
        Gather all samples in the buffer for use in a torch.utils.data.TensorDataset.
        """
        samples = [self.states_2d, self.states_3d, self.state_values,
                   self.expert_actions_r, self.expert_actions_t,
                   self.actions_r, self.actions_t, self.actions_logprob]

        samples += self.get_returns_and_advantages()

        return [catcat(sample) for sample in samples]

    def clear(self):
        """
        Clears the buffer and all its trajectory lists.
        """
        self.count = 0
        self.states_2d.clear()
        self.states_3d.clear()
        self.state_values.clear()
        self.rewards.clear()

        self.expert_actions_r.clear()
        self.expert_actions_t.clear()

        self.actions_r.clear()
        self.actions_t.clear()
        self.actions_logprob.clear()
