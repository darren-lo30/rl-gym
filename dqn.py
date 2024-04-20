import copy
import random

from itertools import count
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from agent import Agent


class DQNNet(nn.Module): 
  def __init__(self, num_states, num_actions):
    super(DQNNet, self).__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, num_actions)
    )

  def forward(self, s):
    return self.net(s)

class DQN(Agent): 
  def __init__(self, env, device, Q_model):
    self.env = env

    # Hyperparameters
    self.eps_start = 0.9
    self.eps_end = 0.05
    self.eps_decay = 1000

    self.batch_size = 256
    self.target_update_freq = 100
    self.gamma = 0.99
    self.replay_memory = deque(maxlen=10_000)
    self.lr = 1e-4
    self.num_episodes = 1000

    self.num_action = 0

    # Initialize target model and Q model
    # Initially they should be identical
    self.Q_model = Q_model.to(device)
    self.target = copy.deepcopy(self.Q_model).to(device)
    self.target.load_state_dict(self.Q_model.state_dict())

    self.optim = torch.optim.AdamW(self.Q_model.parameters(), self.lr, amsgrad=True)
    self.device = device

  def update_target(self):
    self.target.load_state_dict(self.Q_model.state_dict())

  def get_eps(self):
    return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.num_action / self.eps_decay)

  def act(self, state):
    return self.Q_model(state).argmax(dim = 1).item()

  def get_action(self, state):
    self.num_action += 1
    sample = random.random()
    eps = self.get_eps()
    
    if sample > eps:
      with torch.no_grad():
        # Take highest probability based on policy
        res = self.Q_model(state.view(1, -1)).argmax(1)
        return res
    else:
      # Sample random action
      return torch.tensor([self.env.action_space.sample()], device=self.device)

  def optim_Q(self):
    if len(self.replay_memory) < self.batch_size:
      return
    
    batch = random.sample(self.replay_memory, self.batch_size)
    split_batch = [*zip(*batch)]
    state = torch.stack(list(split_batch[0]))
    action = torch.stack(list(split_batch[1]))
    reward = torch.tensor(list(split_batch[2]), device=self.device).view(-1, 1)
    next_state = list(split_batch[3])
    
    # Get calculated Q(s, a)
    curr_state_action = self.Q_model(state).gather(1, action)

    # Compute the target reward Q(s, a) = r(s, a) + gamma * max_a Q_target(s', a)
    non_final_mask = torch.tensor(list(map(lambda ns: ns is not None, next_state)), device=self.device, dtype=torch.bool)
    non_final_next_state = torch.stack([ns for ns in next_state if ns is not None])
    target_state_action = torch.zeros((state.shape[0], 1), dtype=torch.float32, device=self.device)
    with torch.no_grad():
      target_state_action[non_final_mask] = self.target(non_final_next_state).max(1).values.view(-1, 1)
    target_state_action_sum = self.gamma * target_state_action + reward

    loss = nn.functional.smooth_l1_loss(curr_state_action, target_state_action_sum, reduction='mean')
    self.optim.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(self.Q_model.parameters(), 100)
    self.optim.step()
    
  def train(self):
    episode_lens = []
    for episode in range(self.num_episodes):
      if episode % 10 == 0: 
        print(f"Training episode {episode}")
        print(f"Episode takes on average {np.mean(episode_lens[-10:])} steps")

      state = self.env.reset()
      state = torch.tensor(state).to(self.device)

      done = False
      for t in count():
        action = self.get_action(state)
        observation, reward, terminated, truncated = self.env.step(action.item())

        done = terminated or truncated

        if terminated:
          next_state = None
        else:
          next_state = torch.tensor(observation, device=self.device)

        self.replay_memory.append((state, action, reward, next_state))
        state = next_state

        self.optim_Q()

        # Hard updates scheme, not good
        # Update target to Q
        # if self.num_action % self.target_update_freq == 0:
        #   print("HARD UPDATE")
        #   self.update_target()

        # Soft update scheme
        TAU = 0.005
        target_state_dict = self.target.state_dict()
        q_model_dict = self.Q_model.state_dict()
        for key in q_model_dict:
            target_state_dict[key] = q_model_dict[key]*TAU + target_state_dict[key]*(1-TAU)
        self.target.load_state_dict(target_state_dict)

        if done:
          episode_lens.append(t)
          break

  def save(self, file):
    torch.save(self.Q_model.state_dict(), file)
  
  def load(self, file):
    self.Q_model.load_state_dict(torch.load(file))