import torch
import time
import gym

def get_device():
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run(model):

  env = gym.make("CartPole-v1")

  state = env.reset()
  done = False
  while not done:
    state = torch.tensor(state, device=get_device())
    with torch.no_grad():
      action = torch.argmax(model(state))
    state, _, terminated, truncated = env.step(action.item())
    done = terminated or truncated
    if not done:
      env.render()
    time.sleep(0.01)

  env.close()