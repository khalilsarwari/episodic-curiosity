from dotmap import DotMap
from stable_baselines3 import PPO

# Configuration for PPO on Breakout

config = DotMap()
config.agent = PPO
config.environment = 'Breakout-v0'
config.policy_model = 'MlpPolicy'
config.tb_subdir = "ppo_breakout"
config.total_timesteps = 10000000