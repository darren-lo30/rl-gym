import torch
from torch import nn
import gym
from utils import *
from agent import *
from itertools import count, chain
import numpy as np


class PPO(Agent):
  def __init__(self, env, policy_net, value_net, device=get_device(), save_file='./data/ppo'):
    super().__init__(env, device)
    self.policy_net = policy_net
    self.value_net = value_net
    self.save_file = save_file

    # Hyperparameters
    lr = 1e-4
    self.clip_epsilon = 0.2
    self.gamma = 0.99
    self.num_train_epochs = 2000
    self.num_optim_epochs = 10
    self.buffer_size = 2048 
    self.batch_size = 256
    self.advantage_coef = 1
    self.entropy_coef = 1e-4
    self.optim = torch.optim.AdamW(chain(policy_net.parameters(), value_net.parameters()), lr, amsgrad=True)
    
    self.batch_episode_lens = []

  def compute_advantage(self, state, next_state, reward):
    v_curr = self.value_net(state)
    if next_state is None:
      v_next = 0
    else:
      v_next = self.value_net(next_state)
    
    target = reward + self.gamma * v_next
    advantage = target - v_curr
    return advantage

  def sample_action(self, state):
    p_actions = self.policy_net(state)
    choice = torch.multinomial(p_actions, 1, replacement=True)
    p_action = p_actions[choice.item()].reshape(1)
    return p_action, choice
  
  def get_p_actions(self, states, actions):
    p_actions_all = self.policy_net(states)
    p_actions = torch.gather(p_actions_all, 1, actions.reshape(-1, 1))
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
  
  def optim_steps(self, states, actions, p_actions, advantages):
    dataset = torch.utils.data.TensorDataset(states, actions, p_actions, advantages)
    states_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    for _ in range(self.num_optim_epochs):
      for states_batch, actions_batch, p_actions_batch, advantages_batch in states_dataloader:
        self.optim.zero_grad()
        # probability under updated recent policy
        new_p_actions = self.get_p_actions(states_batch, actions_batch)
        ratio = new_p_actions / p_actions_batch

        unclipped_objective = ratio * advantages_batch
        clipped_ratio = torch.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) 

        clipped_objective = clipped_ratio * advantages_batch

        actor_objective = torch.min(unclipped_objective, clipped_objective).sum()
        advantage_loss = torch.square(advantages_batch).sum()

        entropy = -torch.sum(p_actions_batch * torch.log(p_actions_batch + 1e-10))

        loss = actor_objective + self.advantage_coef * advantage_loss - self.entropy_coef * entropy
        loss.backward()
        
        self.optim.step()
      
  def collect_buffer(self):
    t = 0

    states = []
    advantages = []
    p_actions = []
    actions = []

    state = torch.tensor(self.env.reset(), device=self.device)
    done = False
    while len(states) < self.buffer_size:
      if done:
        state = torch.tensor(self.env.reset(), device=self.device)
        done = False
        self.batch_episode_lens.append(t)
        t = 0

      t += 1
      p_action, action = self.sample_action(state)
      next_state, reward, terminated, truncated = self.env_step(action)
      advantage = self.compute_advantage(state, next_state, reward)

      states.append(state.reshape(1, -1))
      p_actions.append(p_action.detach())
      advantages.append(advantage.detach())
      actions.append(action)

      done = terminated or truncated
      if not done:
        state = next_state

    return torch.cat(states, dim = 0), torch.cat(advantages), torch.cat(p_actions), torch.cat(actions)  

  def train(self):
    episode_lens = []

    for epoch in range(self.num_train_epochs):
      states, advantages, p_actions, actions = self.collect_buffer()
      self.optim_steps(states, actions, p_actions, advantages)

      # Store average episode length
      mean_episode_len = np.mean(self.batch_episode_lens)
      episode_lens.append(mean_episode_len)
      print(f'Training epoch {epoch}. Lasted on average {mean_episode_len}')
      self.batch_episode_lens = []
    
    return episode_lens
      
if __name__ == "__main__":
  device = get_device()
  env = gym.make("CartPole-v1")
  num_states, num_actions = get_num_states_actions_discrete(env)

  policy_net = PolicyNet(num_states, num_actions).to(device=device)
  value_net = ValueNet(num_states).to(device=device)
  r = PPO(env, policy_net, value_net)
  episode_lens = train_save_run(r)
