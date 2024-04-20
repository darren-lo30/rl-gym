import gym
import torch
from utils import get_device
import time

# A base class for RL agents
class Agent():
  def __init__(self, env, device):
    self.env = env
    self.device = device
  
  # Train the agent simulating num_episodes
  def train(self):
    raise NotImplementedError()

  # Get the optimal action at a state
  # This should handle both continuous and discrete cases depending on the environment
  def act(self, state):
    raise NotImplementedError()

  # Save the agent's model to a file
  def save(self):
    raise NotImplementedError()

  # Load the agent's model from a file
  def load(self):
    raise NotImplementedError()
  
  def env_step(self, action):
    if torch.is_tensor(action) and torch.squeeze(action).shape[0] == 1:
      action = action.item()

    next_state, reward, terminated, truncated, _ = self.env.step(action)
    if next_state is not None:
      next_state = torch.tensor(next_state, device=self.device)
    reward = torch.tensor(reward, device=self.device).reshape(1)

    return next_state, reward, terminated, truncated


# Simulate the agent in the environment
def run(agent):
  env = agent.env
  state, _ = env.reset()
  done = False
  while not done:
    state = torch.tensor(state, device=agent.device)
    with torch.no_grad():
      action = agent.act(state)
    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    if not done:
      env.render()
    time.sleep(0.01)

  env.close()

# Train, save then run an agent
  
def train_run(agent):
  hist = agent.train()
  run(agent)
  return hist

def train_save_run(agent):
  hist = agent.train()
  agent.save()
  run(agent)
  return hist

# Load and run an agent
def load_and_run(agent):
  agent.load()
  run(agent)