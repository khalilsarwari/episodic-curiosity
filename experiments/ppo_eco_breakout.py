from dotmap import DotMap
from stable_baselines3 import PPO_ECO

# Configuration for PPO with Episodic Curiosity on Breakout

config = DotMap()
config.agent = PPO_ECO
config.environment = 'Breakout-v0'
config.policy_model = 'CnnPolicy'
config.tb_subdir = "ppo_eco_breakout"
config.total_timesteps = 10000000