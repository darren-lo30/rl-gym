from ppo import PPO, PPOContinuousPolicyNet
from utils import ValueNet, get_num_states_actions_continuous, get_num_states_actions_discrete
import gym

def is_continuous_action(env):
  return type(env.action_space) == gym.spaces.box.Box
  
def get_PPO(env, device):
  # Can handle both discrete and continuous environments
  if is_continuous_action(env):
    num_states, num_actions = get_num_states_actions_continuous(env)
  else:
    num_states, num_actions = get_num_states_actions_discrete(env)
    
  policy_net = PPOContinuousPolicyNet(num_states, num_actions)
  value_net = ValueNet(num_states)
  PPO(env, policy_net, value_net, device=device, save_file='./data/bipedal/ppo-bipedal')

def get_model(model_type, env, device):
  return model_mapping[model_type](env, device)

model_mapping = {
  'ppo': get_PPO, 
}

def get_model_names():
  return model_mapping.keys()