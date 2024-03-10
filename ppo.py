import torch
from torch import nn
import gym
from utils import *
from agent import *
from itertools import count, chain
import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset


class PPOPolicyNet(nn.Module): 
  def __init__(self, num_states, num_actions):
    super(PPOPolicyNet, self).__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(),
      torch.nn.Linear(128, 128), torch.nn.ReLU(), 
      torch.nn.Linear(128, num_actions), torch.nn.Softmax(dim=0)
    )

  def forward(self, s):
    p = self.net(s)
    return torch.distributions.Categorical(p)
  
class ExperienceDataset(Dataset):
  def __init__(self, buffer):
    self.buffer = buffer

  def __len__(self):
    return len(self.buffer)
  
  def __getitem__(self, index):
    return self.buffer[index]

class PPO(Agent):
  def __init__(self, env, policy_net, value_net, device=get_device(), save_file='./data/ppo'):
    super().__init__(env, device)
    self.policy_net = policy_net
    self.value_net = value_net
    self.save_file = save_file

    # Hyperparameters
    lr = 0.0003
    self.clip_epsilon = 0.2
    self.gamma = 0.99
    self.num_train_epochs = 300
    self.num_optim_epochs = 5
    self.buffer_size = 128
    self.batch_size = 16
    self.gae_lambda=0.95
    self.advantage_coef = 0.5
    self.entropy_coef = 0
    self.optim = torch.optim.AdamW(chain(policy_net.parameters(), value_net.parameters()), lr)
    self.batch_episode_lens = []

  def sample_action(self, state):
    dist = self.policy_net(state)
    action = dist.sample()
    p_action = dist.log_prob(action)
    return p_action, action
  
  def get_p_actions(self, states, actions):
    dists = self.policy_net(states)
    p_actions = dists.log_prob(actions.reshape(-1))
    return p_actions
    
  def act(self, state):
    _, action = self.sample_action(state)
    return action.item()

  def save(self):
    torch.save({
      'policy': self.policy_net.state_dict(),
      'value': self.value_net.state_dict()
    }, self.save_file)

  def load(self):
    data = torch.load(self.save_file)
    self.policy_net.load_state_dict(data['policy'])
    self.value_net.load_state_dict(data['value'])
  
  def optim_steps(self, buffer):
    states_dataloader = torch.utils.data.DataLoader(ExperienceDataset(buffer), batch_size=self.batch_size, shuffle=True)
    
    for _ in range(self.num_optim_epochs):
      for batch in states_dataloader:
        states, actions, p_actions, advantages, values = batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # probability under updated recent policy
        new_p_actions = self.get_p_actions(states, actions)
        ratio = torch.exp(new_p_actions - p_actions)

        unclipped_objective = ratio * advantages
        clipped_ratio = torch.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) 

        clipped_objective = clipped_ratio * advantages

        actor_objective = torch.min(unclipped_objective, clipped_objective)
        returns = advantages + values 
        new_values = self.value_net(states)

        value_loss = torch.nn.functional.mse_loss(new_values, returns, reduction='none')
        entropy = -new_p_actions.exp() * new_p_actions

        loss = -actor_objective + self.advantage_coef * value_loss 

        self.optim.zero_grad()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 40)
        nn.utils.clip_grad_norm_(self.value_net.parameters(), 40)
        loss.mean().backward()
        self.optim.step()
      # self.scheduler.step()
  
  def compute_gae(self, states, rewards):
    gae = [0 for i in range(len(rewards))]
    values = [self.value_net(state) for state in states]
    for t in range(len(rewards)-1):
      discount = 1
      a_t = 0
      for k in range(t, len(rewards)-1):
        a_t += discount*(rewards[k] + self.gamma * values[k+1] - values[k])
        discount *= self.gamma*self.gae_lambda
      gae[t] = a_t.detach()
    gae[-1] = torch.tensor([0], device=self.device)
    # returns = 0
    # for (state, reward) in reversed(list(zip(states, rewards))):
    #   returns = returns * self.gamma + reward
    #   advantage = returns - self.value_net(state)
    #   gae.append(advantage.detach())

    return gae
  def collect_buffer(self):
    t = 0

    buffer = []
    while len(buffer) < self.buffer_size:
      done = False
      state, _ = self.env.reset()
      state = torch.tensor(state, device=self.device)
      t = 0
      
      states = []
      actions = []
      rewards = []
      p_actions = []
      values = []

      while not done:
        t += 1
        p_action, action = self.sample_action(state.reshape(-1))
        next_state, reward, terminated, truncated = self.env_step(action)

        value = self.value_net(state)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        p_actions.append(p_action.detach())
        values.append(value.detach())

        done = terminated or truncated
        if not done:
          state = next_state

      self.batch_episode_lens.append(t)
      advantages = self.compute_gae(states, rewards)
      buffer.extend(list(zip(states, actions, p_actions, advantages, values)))

    return buffer[:self.batch_size] 

  def train(self):
    episode_lens = []

    best_episode_len = 0

    for epoch in range(self.num_train_epochs):
      buffer = self.collect_buffer()
      self.optim_steps(buffer)

      # Store average episode length
      mean_episode_len = np.mean(self.batch_episode_lens)
      episode_lens.append(mean_episode_len)
      print(f'Training epoch {epoch}. Lasted on average {mean_episode_len}')

      if mean_episode_len > best_episode_len:
        best_episode_len = mean_episode_len
        self.save()

      self.batch_episode_lens = []
    
    return episode_lens
      
if __name__ == "__main__":
  device = get_device()
  env = gym.make("CartPole-v1")
  num_states, num_actions = get_num_states_actions_discrete(env)

  policy_net = PPOPolicyNet(num_states, num_actions).to(device=device)
  value_net = ValueNet(num_states).to(device=device)
  r = PPO(env, policy_net, value_net)
  episode_lens = train_run(r)
  vis_episodes(episode_lens, './data/ppo')
