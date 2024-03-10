import torch
from torch import nn
import gym
from utils import *
from agent import *
from itertools import count
import numpy as np

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

class BaselineNet(nn.Module):
  def __init__(self, num_states):
    super(BaselineNet, self).__init__() 
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(), 
      torch.nn.Linear(128, 1)
    )
  
  def forward(self, s):
    return self.net(s)

# Reinforce with baseline loss
class ReinforceBaseline(Agent):
  def __init__(self, env, device, policy_net, baseline_net, save_file = './data/reinforce_baseline'):
    super().__init__(env, device)
    self.save_file = save_file
  
    # Hyperparameters
    policy_lr = 1e-4
    baseline_lr = 1e-4
    self.gamma = 0.99
    self.num_episodes = 1000
    
    # Policy model
    self.policy_net = policy_net.to(device=device)
    self.baseline_net = baseline_net.to(device=device)
    self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=policy_lr, amsgrad=True)
    self.baseline_optim = torch.optim.AdamW(self.baseline_net.parameters(), lr=baseline_lr, amsgrad=True)

  def policy_update(self, p_actions, G, baselines):
    # Update baseline
    sv_loss = torch.nn.functional.mse_loss(G, baselines)
    self.baseline_optim.zero_grad()
    sv_loss.backward()

    self.baseline_optim.step()

    # Update policy
    # We want to maximize the P * R, so minimize -P * R
    rewards = (G - baselines.detach())
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
    action_grad = -torch.log(p_actions) * rewards
    action_grad = action_grad.sum()
    
    self.policy_optim.zero_grad()
    action_grad.backward()
    # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 40)

    self.policy_optim.step()
    
  def sample_action(self, state):
    p_actions = self.policy_net(state)
    action = torch.multinomial(p_actions, 1, replacement=True)
    p_action = p_actions[action.item()].reshape(1)
    return p_action, action

  def act(self, state):
    _, action = self.sample_action(state)
    return action.item()
  
  def compute_G(self, rewards):
    pows = torch.pow(self.gamma, torch.arange(0, rewards.shape[0], device=self.device))
    # If you remove / pows, this will match Sutton & Barto's formulation where each gradient is multiplied by gamma^t 
    G = torch.flip(torch.cumsum(torch.flip(rewards * pows, dims = [0]), dim = 0), dims = [0]) / pows
    return G

  def train(self):
    episode_lens = []
    for episode in range(self.num_episodes):
      state = self.env.reset()
      state = torch.tensor(state, device=self.device)
        
      rewards = []
      p_actions = []
      baselines = []
      done = False
      # Collect data
      for t in count():
        p_action, action = self.sample_action(state)
        next_state, reward, terminated, truncated = self.env_step(action.item())
        rewards.append(reward)
        p_actions.append(p_action)

        baseline = self.baseline_net(state)
        baselines.append(baseline)

        done = terminated or truncated
        if not done:
          state = next_state
        else:
          episode_lens.append(t)
          break
      rewards = torch.cat(rewards)
      p_actions = torch.cat(p_actions)
      baselines = torch.cat(baselines)
      G = self.compute_G(rewards)
      # Update policy gradient  
      self.policy_update(p_actions, G, baselines)

      if episode % 10 == 0:
        print(f"Simulating episode {episode}. Lasted on average {np.mean(episode_lens[-10:])}")
    
    return episode_lens
  
  def save(self):
    torch.save({
      'policy': self.policy_net.state_dict(),
      'baseline': self.baseline_net.state_dict()
    }, self.save_file)

  def load(self):
    data = torch.load(self.save_file)
    self.policy_net.load_state_dict(data['policy'])
    self.baseline_net.load_state_dict(data['baseline'])

    
if __name__ == "__main__":
  device = get_device()
  env = gym.make("CartPole-v1")
  # Reinforce with baseline loss
  num_states, num_actions = get_num_states_actions_discrete(env)

  policy_net = PolicyNet(num_states, num_actions)
  baseline_net = BaselineNet(num_states)

  r = ReinforceBaseline(env, device, policy_net, baseline_net, save_file='./data/reinforce_baseline_acrobot')
  episode_lens = train_save_run(r)
  vis_episodes(episode_lens, './data/reinforce_baseline')