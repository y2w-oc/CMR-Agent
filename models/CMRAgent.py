import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import sys
import cv2
import scipy.io as scio
from torch.distributions import Categorical

from .PointNN import ConvBNReLURes1D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CMRAgent(nn.Module):
    def __init__(self, config):
        super(CMRAgent, self).__init__()
        self.config = config

        f = config.embed_dim

        # 2D-3D state fusion head
        self.state_3d_embed = nn.ModuleList()
        self.state_3d_embed.append(ConvBNReLURes1D(5, f))
        self.state_3d_embed.append(ConvBNReLURes1D(2*f, f))
        self.state_3d_embed.append(ConvBNReLURes1D(2*f, f))
        self.state_3d_embed.append(ConvBNReLURes1D(2*f, 2*f))

        H = config.image_H // 8
        W = config.image_W // 8

        self.state_2d_embed = nn.Sequential(nn.Conv2d(2*f, 2*f, kernel_size=(3, 3), padding=(1, 1), stride=1),
                                            nn.BatchNorm2d(2*f),
                                            nn.LeakyReLU(inplace=True),
                                            nn.Conv2d(2*f, 2*f, kernel_size=(3, 3), padding=(1, 1), stride=1),
                                            nn.LeakyReLU(inplace=True),
                                            nn.AvgPool2d((2, 2), stride=(2, 2)),
                                            nn.Conv2d(2*f, 2*f, kernel_size=(3, 3), padding=(1, 1), stride=1),
                                            nn.BatchNorm2d(2*f),
                                            nn.LeakyReLU(inplace=True),
                                            nn.Conv2d(2*f, 2*f, kernel_size=(3, 3), padding=(1, 1), stride=1),
                                            nn.LeakyReLU(inplace=True),
                                            nn.AvgPool2d((2, 2), stride=(2, 2)),
                                            nn.Conv2d(2*f, 2*f, kernel_size=(3, 3), padding=(1, 1), stride=1),
                                            nn.BatchNorm2d(2*f),
                                            nn.LeakyReLU(inplace=True),
                                            nn.Conv2d(2*f, 2*f, kernel_size=(3, 3), padding=(1, 1), stride=1),
                                            nn.LeakyReLU(inplace=True),
                                            nn.AvgPool2d((2, 2), stride=(2, 2)),
                                            nn.Conv2d(2*f, 2*f, kernel_size=(3, 3), padding=(1, 1), stride=1),
                                            nn.BatchNorm2d(2*f),
                                            nn.LeakyReLU(inplace=True),
                                            nn.Conv2d(2*f, 2*f, kernel_size=(3, 3), padding=(1, 1), stride=1),
                                            nn.LeakyReLU(inplace=True),
                                            nn.AvgPool2d((H, W), stride=1),
                                            nn.Conv2d(2*f, 2*f, kernel_size=(1, 1), padding=(0, 0), stride=1),
                                            nn.LeakyReLU(inplace=True),
                                            nn.Conv2d(2*f, 2*f, kernel_size=(1, 1), padding=(0, 0), stride=1))

        # actor-critic head
        if config.is_6_DoF:
            self.degree_r = 3
            self.degree_t = 3
        else:
            self.degree_r = 1
            self.degree_t = 2

        self.policy_r = nn.Sequential(nn.Linear(4 * f, 4 * f),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Linear(4 * f, 4 * f),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Linear(4 * f, self.degree_r * config.num_steps))

        self.policy_t = nn.Sequential(nn.Linear(4 * f, 4 * f),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Linear(4 * f, 4 * f),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Linear(4 * f, self.degree_t * config.num_steps))

        self.value = nn.Sequential(nn.Linear(4 * f, f),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Linear(f, f),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Linear(f, 1))

    def forward(self, state_2d, state_3d):
        embed_2d = self.state_2d_embed(state_2d)
        embed_2d = embed_2d.view(embed_2d.shape[0], -1)

        embed_3d = state_3d
        step = 0
        for layer in self.state_3d_embed:
            feat_3d = layer(embed_3d)
            embed_3d = torch.max(feat_3d, dim=2, keepdim=True)[0]
            if step < len(self.state_3d_embed)-1:
                embed_3d = embed_3d.repeat(1, 1, feat_3d.shape[2])
                embed_3d = torch.cat([feat_3d, embed_3d], dim=1)
            step = step + 1
        embed_3d = embed_3d.view(embed_3d.shape[0], -1)

        state_embedding = torch.cat([embed_2d, embed_3d], dim=1)
        # state_embedding = embed_3d

        action_r_logits = self.policy_r(state_embedding)
        action_t_logits = self.policy_t(state_embedding)

        action_r_logits = action_r_logits.view(action_r_logits.shape[0], self.degree_r, self.config.num_steps)
        action_t_logits = action_t_logits.view(action_t_logits.shape[0], self.degree_t, self.config.num_steps)

        value = self.value(state_embedding)
        value = value.unsqueeze(-1)

        return action_r_logits, action_t_logits, value

    @staticmethod
    def action_from_logits(r_logits, t_logits, deterministic=False):
        distribution_r = Categorical(logits=r_logits)
        distribution_t = Categorical(logits=t_logits)
        if deterministic:
            action_t = torch.argmax(distribution_t.probs, dim=-1)
            action_r = torch.argmax(distribution_r.probs, dim=-1)
        else:
            action_t = distribution_t.sample()
            action_r = distribution_r.sample()
        return action_r, action_t

    @staticmethod
    def action_logprob_and_entropy(r_logits, t_logits, action_r, action_t):

        distribution_r = Categorical(logits=r_logits)
        distribution_t = Categorical(logits=t_logits)

        logprob_r = distribution_r.log_prob(action_r)
        logprob_t = distribution_t.log_prob(action_t)

        entropy_r = distribution_r.entropy()
        entropy_t = distribution_t.entropy()

        logprob = torch.cat([logprob_r, logprob_t], dim=1)
        entropy = torch.cat([entropy_r, entropy_t], dim=1)

        return logprob, entropy
