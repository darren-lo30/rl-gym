import torch
from torch import nn

class ReinforceNet(nn.Module): 
  def __init__(self, num_states, num_actions):
    super(ReinforceNet, self).__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, num_actions)
    )

  def forward(self, s):
    return self.net(s)


class CartPoleReinforce():
  def __init__():