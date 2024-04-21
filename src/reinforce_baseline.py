import torch
from itertools import count
import numpy as np

from agent import Agent

# Reinforce with baseline loss
class ReinforceBaseline(Agent):
  def __init__(self, config, env, device, policy_net, baseline_net):
    super().__init__(config, env, device)

    
    # Policy model
    self.policy_net = policy_net.to(device=device)
    self.baseline_net = baseline_net.to(device=device)
    self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=self.config.policy_lr, amsgrad=True)
    self.baseline_optim = torch.optim.AdamW(self.baseline_net.parameters(), lr=self.config.baseline_lr, amsgrad=True)

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
    pows = torch.pow(self.config.gamma, torch.arange(0, rewards.shape[0], device=self.device))
    # If you remove / pows, this will match Sutton & Barto's formulation where each gradient is multiplied by gamma^t 
    G = torch.flip(torch.cumsum(torch.flip(rewards * pows, dims = [0]), dim = 0), dims = [0]) / pows
    return G

  def train(self):
    episode_lens = []
    for episode in range(self.config.num_episodes):
      state, _ = self.env.reset()
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
  
  def save(self, file):
    torch.save({
      'policy': self.policy_net.state_dict(),
      'baseline': self.baseline_net.state_dict()
    }, file)

  def load(self, file):
    data = torch.load(file)
    self.policy_net.load_state_dict(data['policy'])
    self.baseline_net.load_state_dict(data['baseline'])

  