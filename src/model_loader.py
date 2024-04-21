from ppo import PPO, PPOContinuousPolicyNet, PPODiscretePolicyNet
from dqn import DQN, DQNNet
from a2c import ActorCritic
from reinforce import Reinforce
from reinforce_baseline import ReinforceBaseline
from utils import PolicyNet, ValueNet, get_num_states_actions_continuous, get_num_states_actions_discrete
import gym

def is_continuous_action(env):
  return type(env.action_space) == gym.spaces.box.Box

def get_ppo(config, env, device):
  # Can handle both discrete and continuous environments
  if is_continuous_action(env):
    num_states, num_actions = get_num_states_actions_continuous(env)
    policy_net = PPOContinuousPolicyNet(num_states, num_actions)
  else:
    num_states, num_actions = get_num_states_actions_discrete(env)
    policy_net = PPODiscretePolicyNet(num_states, num_actions)
    
  value_net = ValueNet(num_states)
  return PPO(config, env, policy_net, value_net, device=device)

def get_dqn(config, env, device):
  if is_continuous_action(env):
    NotImplementedError()
  else:
    num_states, num_actions = get_num_states_actions_discrete(env)
    net = DQNNet(num_states, num_actions)
    return DQN(config, env, device, net)

def get_a2c(config, env, device):
  if is_continuous_action(env):
    NotImplementedError()
  else:
    num_states, num_actions = get_num_states_actions_discrete(env)
    actor = PolicyNet(num_states, num_actions)
    critic = ValueNet(num_states)
    
    return ActorCritic(config, env, device, actor, critic)

def get_reinforce(config, env, device):
  if is_continuous_action(env):
    NotImplementedError()
  else:    
    num_states, num_actions = get_num_states_actions_discrete(env)
    policy = PolicyNet(num_states, num_actions)
    return Reinforce(config, env, device, policy)

def get_reinforce_baseline(config, env, device):
  if is_continuous_action(env):
    NotImplementedError()
  else:    
    num_states, num_actions = get_num_states_actions_discrete(env)
    policy = PolicyNet(num_states, num_actions)
    baseline = ValueNet(num_states)
    return ReinforceBaseline(config, env, device, policy, baseline)
  
def get_model(model_type, config, env, device):
  return model_mapping[model_type](config, env, device)

model_mapping = {
  'ppo': get_ppo,
  'dqn': get_dqn,
  'a2c': get_a2c,
  'reinforce': get_reinforce,
  'reinforce_baseline': get_reinforce_baseline
}

def get_model_names():
  return model_mapping.keys()