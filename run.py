import argparse
import importlib
import os
import subprocess
from dotmap import DotMap
import gym
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common import logger
from stable_baselines3.dummy import DummyEnvWrapper

import utils

# Regular Runner, no episodic curiosity

def main(config):
    if config.atari_wrapper:
        train_env = make_atari_env(config.environment, n_envs=config.workers)
    else:
        train_env = make_vec_env(config.environment, n_envs=config.workers)
    train_env = DummyEnvWrapper(train_env, config.add_stoch)

    tb_dir = os.path.join(config.log_dir, config.tb_subdir)
    model = config.agent(config.policy_model, train_env, config, verbose=config.verbose, tensorboard_log=tb_dir)

    model.learn(total_timesteps=config.total_timesteps)

    if config.final_vis:
        env = gym.make(config.environment)
        obs = env.reset()
        for i in range(config.final_vis_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()

        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="episodic-curiosity")
    parser.add_argument("-exp", "--experiment", type=str, required=True, help='name of config file in experiment folder')
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument('--tb_port', action="store", type=int, default=6006, help="tensorboard port")

    # per run args
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--final_vis", type=bool, default=False)
    parser.add_argument("--final_vis_steps", type=int, default=1000)

    args = parser.parse_args()
    config = importlib.import_module('experiments.{}'.format(args.experiment)).config

    # update config dotmap
    config = config.toDict()
    config.update(vars(args))
    config = DotMap(config)

    # kill existing tensorboard processes on port (in order to refresh)
    utils.kill_processes_on_port(config.tb_port)

    env = dict(os.environ)   # Make a copy of the current environment
    subprocess.Popen('tensorboard --host 0.0.0.0 --port {} --logdir ./{}'.format(config.tb_port, config.log_dir), env=env, shell=True)

    main(config)