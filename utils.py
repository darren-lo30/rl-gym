import torch
import time
import gym
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