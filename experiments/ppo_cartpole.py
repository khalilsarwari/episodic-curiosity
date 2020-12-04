from dotmap import DotMap
from stable_baselines3 import PPO

# Configuration for PPO on Cartpole

config = DotMap()
config.agent = PPO
config.environment = 'CartPole-v1'
config.policy_model = 'CnnPolicy'
config.tb_subdir = "ppo_cartpole"
config.total_timesteps = 2e6