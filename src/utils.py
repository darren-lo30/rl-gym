import torch
from torch import nn
import matplotlib.pyplot as plt

def get_device():
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def vis_episodes(episode_lens, save_file):
  plt.plot(episode_lens)
  plt.ylabel('Episode length')
  plt.xlabel('Episode number')
  plt.savefig(save_file)

def get_num_states_actions_discrete(env):
  num_states = env.observation_space.shape[0]
  num_actions = env.action_space.n

  return num_states, num_actions

def get_num_states_actions_continuous(env):
  num_states = env.observation_space.shape[0]
  num_actions = env.action_space.shape[0]

  return num_states, num_actions

class PolicyNet(nn.Module): 
  def __init__(self, num_states, num_actions):
    super(PolicyNet, self).__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(), 
      torch.nn.Linear(128, num_actions), torch.nn.Softmax(dim=0)
    )

  def forward(self, s):
    return self.net(s)

class ValueNet(nn.Module):
  def __init__(self, num_states):
    super(ValueNet, self).__init__() 
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(), 
      torch.nn.Linear(128, 1)
    )
  
  def forward(self, s):
    return self.net(s)
