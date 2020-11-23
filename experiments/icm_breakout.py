from dotmap import DotMap
from stable_baselines3 import ICM

# Configuration for ICM on Breakout

config = DotMap()
config.agent = ICM
config.environment = 'Breakout-v0'
config.policy_model = 'MlpPolicy'
config.tb_subdir = "icm_breakout"
config.total_timesteps = 10000000