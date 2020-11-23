from dotmap import DotMap
from stable_baselines3 import ICM

# Configuration for ICM on Montezuma's Revenge

config = DotMap()
config.agent = ICM
config.environment = 'MontezumaRevenge-v0'
config.policy_model = 'MlpPolicy'
config.tb_subdir = "icm_montezuma"
config.total_timesteps = 10000000
