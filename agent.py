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
  def train(self, num_episodes):
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

# Simulate the agent in the environment
def run(agent):
  env = agent.env
  state = env.reset()
  done = False
  while not done:
    state = torch.tensor(state, device=get_device())
    with torch.no_grad():
      action = agent.act(state)
    state, _, terminated, truncated = env.step(action)
    done = terminated or truncated
    if not done:
      env.render()
    time.sleep(0.01)

  env.close()

# Train, save then run an agent
def train_save_run(agent, num_episodes=1000):
  hist = agent.train(num_episodes)
  agent.save()
  run(agent)
  return hist

# Load and run an agent
def load_and_run(agent):
  agent.load()
  run(agent)