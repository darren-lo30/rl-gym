import torch
from torch import nn
import gym
from utils import *
from itertools import count
import numpy as np
import agent

class ReinforceNet(nn.Module): 
  def __init__(self, num_states, num_actions):
    super(ReinforceNet, self).__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(), 
      torch.nn.Linear(128, num_actions), torch.nn.Softmax(dim=0)
    )

  def forward(self, s):
    return self.net(s)

class Reinforce(agent.Agent):
  def __init__(self, env, device, policy_net):
    super().__init__(env, device)
  
    # Hyperparameters
    lr = 1e-4
    self.gamma = 0.99
    self.num_episodes = 1000

    # Policy model
    self.policy_net = policy_net.to(device)
    self.optim = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

  def policy_update(self, p_actions, G):
    # We want to maximize the P * R, so minimize -P * R
    action_grad = -torch.log(p_actions) * G
    action_grad = action_grad.sum()
    
    self.optim.zero_grad()
    action_grad.backward()
    # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 40)

    self.optim.step()

  def sample_action(self, state):
    p_actions = self.policy_net(state)
    choice = torch.multinomial(p_actions, 1, replacement=True)
    p_action = p_actions[choice.item()].reshape(1)
    return p_action, choice
  
  def act(self, state):
    _, action = self.sample_action(state)
    return action.item()
  
  def compute_G(self, rewards):
    pows = torch.pow(self.gamma, torch.arange(0, rewards.shape[0], device=self.device))
    G = torch.flip(torch.cumsum(torch.flip(rewards * pows, dims = [0]), dim = 0), dims = [0]) / pows
    return G

  def train(self):
    episode_lens = []
    for episode in range(self.num_episodes):
      state = self.env.reset()
      state = torch.tensor(state, device=self.device)
        
      rewards = []
      p_actions = []
      done = False
      # Collect data
      for t in count():
        p_action, action = self.sample_action(state)
        next_state, reward, terminated, truncated = self.env_step(action.item())
        rewards.append(reward)
        p_actions.append(p_action)

        done = terminated or truncated
        if not done:
          state = next_state
        else:
          episode_lens.append(t)
          break
      p_actions = torch.cat(p_actions)
      G = self.compute_G(rewards)
      # Update policy gradient  
      self.policy_update(p_actions, G)

      if episode % 10 == 0:
        print(f"Simulating episode {episode}. Lasted on average {np.mean(episode_lens[-10:])}")

  def save(self):
    torch.save(self.policy_net.state_dict(), './data/reinforce')
  
  def load(self):
    self.policy_net.load_state_dict(torch.load('./data/reinforce'))

    
if __name__ == "__main__":
  env = gym.make("CartPole-v1")
  num_states, num_actions = get_num_states_actions_discrete(env)
  net = ReinforceNet(num_states, num_actions)
  device = get_device()
  r = Reinforce(env, device, net)
  agent.load_and_run(r)