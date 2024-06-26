from itertools import chain
import numpy as np
import torch
from torch import nn

from agent import Agent
from utils import get_device


# Actions are sampled from a multinomial distribution
class PPODiscretePolicyNet(nn.Module):
  def __init__(self, num_states, num_actions):
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, num_actions),
      torch.nn.Softmax(dim=-1),
    )

  def forward(self, s):
    p = self.net(s)
    dist = torch.distributions.Categorical(p)
    return dist

  def sample_action(self, state):
    dist = self(state)
    action = dist.sample()
    p_action = dist.log_prob(action)
    return p_action, action

  def get_p_actions(self, states, actions):
    dists = self(states)
    return dists.log_prob(actions)


# Actions are sampled from a gaussian distribution
class PPOContinuousPolicyNet(nn.Module):
  def __init__(self, num_states, num_actions):
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_states, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU()
    )
    self.mean = torch.nn.Linear(128, num_actions)
    self.log_var = nn.Parameter(torch.tensor([0], dtype=torch.float32))

  def forward(self, s):
    x = self.net(s)
    mu = self.mean(x)

    # Avoid zero log_var
    sigma = torch.exp(self.log_var)

    dist = torch.distributions.Normal(mu, sigma)
    return dist

  def sample_action(self, state):
    dist = self(state)
    action = dist.sample()
    p_action = dist.log_prob(action)
    total_p = torch.sum(p_action, dim=-1)
    return total_p, action

  def get_p_actions(self, states, actions):
    dist = self(states)
    p_action = dist.log_prob(actions)
    total_p = torch.sum(p_action, dim=-1)
    return total_p


class PPO(Agent):
  def __init__(self, config, env, policy_net, value_net, device=get_device()):
    super().__init__(config, env, device)
    self.policy_net = policy_net.to(device)
    self.value_net = value_net.to(device)

    self.optim = torch.optim.Adam(chain(policy_net.parameters(), value_net.parameters()), self.config.lr)
    self.batch_episode_lens = []
    self.batch_rewards = []

  def act(self, state):
    _, action = self.policy_net.sample_action(state)
    return action.cpu()

  def save(self, save_file):
    torch.save(
      {"policy": self.policy_net.state_dict(), "value": self.value_net.state_dict()}, save_file
    )

  def load(self, save_file):
    data = torch.load(save_file)
    self.policy_net.load_state_dict(data["policy"])
    self.value_net.load_state_dict(data["value"])

  def optim_steps(self, buffer):
    states_dataloader = torch.utils.data.DataLoader(
      buffer, batch_size=self.config.batch_size, shuffle=True
    )

    for _ in range(self.config.num_optim_epochs):
      for batch in states_dataloader:
        states, actions, p_actions, advantages, values = batch
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # probability under updated recent policy
        new_p_actions = self.policy_net.get_p_actions(states, actions)
        ratio = new_p_actions.exp() / p_actions.exp()

        unclipped_objective = ratio * advantages
        clipped_ratio = torch.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)

        clipped_objective = clipped_ratio * advantages

        actor_objective = torch.min(unclipped_objective, clipped_objective)

        returns = advantages + torch.squeeze(values)
        new_values = self.value_net(states)

        value_loss = torch.nn.functional.mse_loss(
          torch.squeeze(new_values), returns, reduction="none"
        )

        loss = -actor_objective + self.config.advantage_coef * value_loss

        self.optim.zero_grad()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 2)
        nn.utils.clip_grad_norm_(self.value_net.parameters(), 2)
        loss.mean().backward()
        self.optim.step()
      # self.scheduler.step()

  def compute_gae(self, rewards, values):
    T = len(rewards)
    advantages = torch.zeros(T + 1, dtype=torch.float32, device=self.device)

    for t in reversed(range(T)):
      next_value = 0 if t + 1 >= T else values[t + 1]
      delta_t = rewards[t] + self.config.gamma * next_value - values[t]
      advantages[t] = delta_t + self.config.gae_lambda * self.config.gamma * advantages[t + 1]

    return advantages

  def collect_buffer(self):
    buffer = []
    while len(buffer) < self.config.buffer_size:
      done = False
      state, _ = self.env.reset()
      state = torch.tensor(state, device=self.device)
      t = 0
      total_rewards = 0

      states = []
      actions = []
      rewards = []
      p_actions = []
      values = []

      while not done:
        t += 1
        p_action, action = self.policy_net.sample_action(state.reshape(-1))
        next_state, reward, terminated, truncated = self.env_step(action.cpu())
        done = terminated or truncated

        value = self.value_net(state)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        total_rewards += reward
        p_actions.append(p_action.detach())
        values.append(value.detach())
        done = terminated or truncated
        if not done:
          state = next_state

      self.batch_episode_lens.append(t)
      self.batch_rewards.append(total_rewards.cpu())
      advantages = self.compute_gae(rewards, values)
      buffer.extend(list(zip(states, actions, p_actions, advantages, values)))

    return buffer

  def train(self):
    episode_lens = []
    rewards = []

    # best_reward = -100000

    for epoch in range(self.config.num_train_epochs):
      buffer = self.collect_buffer()
      self.optim_steps(buffer)

      # Store average episode length
      mean_episode_len = np.mean(self.batch_episode_lens)
      mean_rewards = np.mean(self.batch_rewards)
      episode_lens.append(mean_episode_len)
      rewards.append(mean_rewards)
      print(
        f"Training epoch {epoch}. Lasted on average {mean_episode_len}. Reward per run was on average {mean_rewards}."
      )

      # if mean_rewards >= best_reward + 10 and mean_rewards >= 50:
      #   best_reward = mean_rewards
      #   print("Saving model...")
      #   self.save()

      self.batch_episode_lens = []
      self.batch_rewards = []

    return rewards
