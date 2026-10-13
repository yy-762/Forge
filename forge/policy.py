"""Observation-only actor and state-only training critics."""
import torch
from torch import nn
from torch.distributions import Dirichlet, Multinomial


class Actor(nn.Module):
    def __init__(self, obs_dim=9, links=3, hidden=192, packets=10):
        super().__init__()
        self.links, self.packets = links, packets
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.packet_head = nn.Linear(hidden, links)
        self.power_head = nn.Linear(hidden, links)

    def forward(self, observation):
        hidden = self.encoder(observation)
        return self.packet_head(hidden), torch.nn.functional.softplus(self.power_head(hidden)) + 0.3

    def distributions(self, observation):
        logits, concentration = self(observation)
        return Multinomial(self.packets, logits=logits), Dirichlet(concentration)

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        packet_dist, power_dist = self.distributions(observation)
        if deterministic:
            expected = packet_dist.probs * self.packets
            packets = expected.floor()
            remainder = (self.packets - packets.sum(-1)).long()
            order = (expected - packets).argsort(dim=-1, descending=True)
            for row, count in enumerate(remainder.tolist()):
                packets[row, order[row, :count]] += 1
            power = power_dist.concentration / power_dist.concentration.sum(-1, keepdim=True)
        else:
            packets, power = packet_dist.sample(), power_dist.sample()
        log_probability = packet_dist.log_prob(packets) + power_dist.log_prob(power)
        return packets, power, log_probability


class Critic(nn.Module):
    def __init__(self, state_dim=80, hidden=192):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.return_head = nn.Linear(hidden, 1)
        self.risk_head = nn.Linear(hidden, 1)

    def forward(self, state):
        hidden = self.encoder(state)
        return self.return_head(hidden).squeeze(-1), self.risk_head(hidden).squeeze(-1)


def load_actor(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    actor = Actor(**checkpoint['actor_config']).to(device)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    return actor, checkpoint['evaluation_config']
