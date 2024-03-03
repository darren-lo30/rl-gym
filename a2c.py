import torch
from torch import nn
import gym
from utils import get_device, vis_episodes
from itertools import count
import numpy as np

class ActorNet(nn.Module): 
  def __init__(self, num_states, num_actions):
    super(ActorNet, self).__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(), 
      torch.nn.Linear(128, num_actions), torch.nn.Sigmoid(dim=0)
    )

  def forward(self, s):
    return self.net(s)

class CriticNet(nn.Module):
  def __init__(self, num_states):
    super(CriticNet, self).__init__() 
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(), 
      torch.nn.Linear(128, 1)
    )
  
  def forward(self, s):
    return self.net(s)

class ActorCritic():
  def __init__(self, env, device, actor_net, critic_net):
    self.env = env
    self.device = device  
    # Hyperparameters
    lr = 1e-3

    self.gamma = 0.99

    # Policy model
    self.actor_net = actor_net
    self.critic_net = critic_net
    self.actor_optim = torch.optim.AdamW(self.actor_net.parameters(), lr=lr, amsgrad=True)
    self.critic_optim = torch.optim.AdamW(self.critic_net.parameters(), lr=lr, amsgrad=True)

  def policy_update(self, p_actions, advantages, advantages_loss):
    # Compute critic gradients
    self.critic_optim.zero_grad()
    advantages_loss.sum().backward()
    self.critic_optim.step()
    
    # Compute actor gradients    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
    action_grad = -torch.log(p_actions) * advantages
    action_grad = action_grad.sum()
    
    self.actor_optim.zero_grad()
    action_grad.backward()
    # nn.utils.clip_grad_norm_(self.actor_net.parameters(), 40)

    self.actor_optim.step()
    

  def sample_action(self, state):
    p_actions = self.actor_net(state)
    choice = torch.multinomial(p_actions, 1, replacement=True)
    p_action = p_actions[choice.item()].reshape(1)
    return p_action, choice

  def act(self, state):
    _, action = self.sample_action(state)
    return action.item()
  
  def compute_advantage(self, state, next_state, reward):
    v_curr = self.critic_net(state)
    if next_state is None:
      v_next = 0
    else:
      v_next = self.critic_net(next_state)
    
    target = reward + self.gamma * v_next
    advantage = target - v_curr
    loss = torch.nn.functional.mse_loss(target, v_curr).view(1)
    return advantage.detach(), loss

  def train(self, num_episodes):
    episode_lens = []
    for episode in range(num_episodes):
      state = self.env.reset()
      state = torch.tensor(state, device=self.device)
        
      p_actions = []
      advantages = []
      advantages_loss = []
      done = False
      # Collect data
      for t in count():
        p_action, action = self.sample_action(state)
        next_state, reward, terminated, truncated = self.env.step(action.item())
        if next_state is not None:
          next_state = torch.tensor(next_state, device=self.device)
        reward = torch.tensor(reward, device=self.device)

        p_actions.append(p_action)
        advantage, loss = self.compute_advantage(state, next_state, reward)
        advantages.append(advantage)
        advantages_loss.append(loss)

        done = terminated or truncated
        if not done:
          state = next_state
        else:
          episode_lens.append(t)
          break
      p_actions = torch.cat(p_actions)
      advantages = torch.cat(advantages)
      advantages_loss = torch.cat(advantages_loss)
      # Update policy gradient  
      self.policy_update(p_actions, advantages, advantages_loss)

      if episode % 10 == 0:
        print(f"Simulating episode {episode}. Lasted on average {np.mean(episode_lens[-10:])}")
    
    return episode_lens

    
if __name__ == "__main__":
  
  device = get_device()
  env = gym.make("BipedalWalker-v3")
  r = ActorCritic(env, device)
  episode_lens = r.train(3000)
  vis_episodes(episode_lens, './data/a2c')