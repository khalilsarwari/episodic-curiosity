from dotmap import DotMap
from stable_baselines3 import PPO_ICM

# Configuration for PPO with Intrinsic Curiosity on QBert

config = DotMap()
config.agent = PPO_ICM
config.environment = 'Qbert-v0'
config.policy_model = 'CnnPolicy'
config.tb_subdir = "ppo_icm_uncertainty_space_invaders_qbert"
config.total_timesteps = 2e6
config.action_shape = 6
config.ensemble_size = 8
config.atari_wrapper = True
config.add_stoch = True
