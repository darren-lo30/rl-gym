import torch
from torch import nn
import gym
from utils import get_device
from itertools import count
import numpy as np

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


class Reinforce():
  def __init__(self, env, device):
    self.env = env
    self.device = device
    num_states = self.env.observation_space.shape[0]
    num_actions = self.env.action_space.n
  
    # Hyperparameters
    lr = 1e-4
    self.gamma = 0.99

    # Policy model
    self.policy_net = ReinforceNet(num_states, num_actions).to(device=device)
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
  
  def compute_G(self, rewards):
    pows = torch.pow(self.gamma, torch.arange(0, rewards.shape[0], device=self.device))
    G = torch.flip(torch.cumsum(torch.flip(rewards * pows, dims = [0]), dim = 0), dims = [0]) / pows
    return G

  def train(self, num_episodes):
    episode_lens = []
    for episode in range(num_episodes):
      state = self.env.reset()
      state = torch.tensor(state, device=self.device)
        
      rewards = []
      p_actions = []
      done = False
      # Collect data
      for t in count():
        p_action, action = self.sample_action(state)
        next_state, reward, terminated, truncated = self.env.step(action.item())
        rewards.append(reward)
        p_actions.append(p_action)

        done = terminated or truncated
        if not done:
          state = torch.tensor(next_state, device=self.device)
        else:
          episode_lens.append(t)
          break
      rewards = torch.tensor(rewards, device=self.device)
      p_actions = torch.cat(p_actions)
      G = self.compute_G(rewards)
      # Update policy gradient  
      self.policy_update(p_actions, G)

      if episode % 10 == 0:
        print(f"Simulating episode {episode}. Lasted on average {np.mean(episode_lens[-10:])}")


    
if __name__ == "__main__":
  device = get_device()
  env = gym.make("CartPole-v1")
  r = Reinforce(env, device)
  r.train(5000)