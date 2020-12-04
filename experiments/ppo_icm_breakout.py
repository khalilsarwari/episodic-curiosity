from dotmap import DotMap
from stable_baselines3 import PPO_ICM

# Configuration for PPO with Intrinsic Curiosity on Breakout

config = DotMap()
config.agent = PPO_ICM
config.environment = 'Breakout-v0'
config.policy_model = 'MlpPolicy'
config.tb_subdir = "ppo_icm_breakout"
config.total_timesteps = 10000000
config.action_shape = 4