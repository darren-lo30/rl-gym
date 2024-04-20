import argparse

import gym
import torch

from agent import run
from model_loader import get_model, get_model_names
from ppo import PPO
from utils import get_device



def run_load(load_args, device):
  env = gym.make(load_args.env, render_mode='human')
  agent = get_model(load_args.model, env, device)
  agent.load(load_args.save_file)
  run(agent)

def run_train(train_args, device):
  env = gym.make(train_args.env, render_mode=None)
  agent = get_model(train_args.model, env, device)

  agent.train()
  agent.save(train_args.save_file)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="RL-Gym", description="Implementation of RL Gym Algorithms")
  parser.add_argument("--env", type=str, required=True)
  parser.add_argument("--device", choices=['cpu', 'default'], default='default')
  parser.add_argument("--model", choices=get_model_names())
  subparsers = parser.add_subparsers(help="mode", dest="cmd")

  train_parser = subparsers.add_parser("train")  
  train_parser.add_argument("--save_file", type=str)

  load_parser = subparsers.add_parser("load")
  load_parser.add_argument("--save_file", type=str, required=True)
  load_parser.add_argument("--out_video", type=str)

  # Get device
  args = parser.parse_args()
  if(args.device == 'default'):
    device = get_device()
  else:
    device = torch.device("cpu")
  
  if(args.cmd == 'load'):
    run_load(args, device)
  else:
    run_train(args, device)