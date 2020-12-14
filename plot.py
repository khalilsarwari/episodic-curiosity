import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict

plt_dir = 'plots'
if not os.path.exists(plt_dir):
    os.mkdir(plt_dir)

def plot_mean_and_std(means, stds, title, file_name, xaxis='Iterations', yaxis='Extrinsic Reward per Episode', yrange=None):
    """ Plot a mean field and its associated std """
    colors = ['r', 'g', 'b', 'c', 'm']
    lines = []
    for i in range(len(means)):
        mean = np.array(means[i])
        std = np.array(stds[i])
        t = list(range(0, len(mean)*32000, 32000))
        line, = plt.plot(t,mean, color=colors[i])
        lines.append(line)
        plt.fill_between(t, (mean-std), (mean+std), color=colors[i], alpha=.1)
    # plt.legend(lines, ('PPO', 'ICM', 'ECO', 'UM_ICM', 'UM_ECO'))
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Extrinsic Reward")
    plt.title(title)
    plt.savefig(os.path.join(plt_dir, f'{file_name}_plot.png'.replace(' ', '_')))

def plot_result(result, line_name, title, xaxis='Iterations', yaxis='Extrinsic Reward per Episode', yrange=None):
    """ Plot a regular/simple field, a single line """
    t = list(range(len(result)))
    plt.plot(t,result)
    plt.savefig(os.path.join(plt_dir,f'{title}_plot.png'.replace(' ', '_')))

def get_results(file, fields):
    """ Gets the fields from the tb log """
    results = defaultdict(lambda:[])
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag in fields:
                results[v.tag].append(v.simple_value)

    return results

if __name__ == "__main__":
    ####### Space Invaders, DETERMINISTIC #######
    ppo = get_results('logs/space_invaders_deterministic/spaceinvaders_ppo_deterministic', ['rollout/ep_rew_mean', 'rollout/ep_rew_std'])
    ppo_icm = get_results('logs/space_invaders_deterministic/spaceinvaders_ppoicm_deterministic', ['rollout/ep_rew_mean', 'rollout/ep_rew_std'])
    ppo_eco = get_results('logs/space_invaders_deterministic/spaceinvaders_ppoeco_deterministic', ['rollout/ep_rew_mean', 'rollout/ep_rew_std'])
    ppo_icm_unc = get_results('logs/space_invaders_deterministic/spaceinvaders_ppoicmuncertainty_deterministic', ['rollout/ep_rew_mean', 'rollout/ep_rew_std'])
    ppo_eco_unc = get_results('logs/space_invaders_deterministic/spaceinvaders_ppoecouncertainty_deterministic', ['rollout/ep_rew_mean', 'rollout/ep_rew_std'])

    means = [ppo['rollout/ep_rew_mean'], ppo_icm['rollout/ep_rew_mean'], \
             ppo_eco['rollout/ep_rew_mean'], ppo_icm_unc['rollout/ep_rew_mean'], \
             ppo_eco_unc['rollout/ep_rew_mean']]
    stds = [ppo['rollout/ep_rew_std'], ppo_icm['rollout/ep_rew_std'], \
            ppo_eco['rollout/ep_rew_std'], ppo_icm_unc['rollout/ep_rew_std'], \
            ppo_eco_unc['rollout/ep_rew_std']]

    plot_mean_and_std(means, stds, "Space Invaders, Deterministic", file_name="Space-Invaders-Deterministic")



