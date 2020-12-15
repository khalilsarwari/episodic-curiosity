## Episodic Curiosity

The underlying codebase is [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3). See OLD_README.md for its original info.


# Getting Started

`python -m pip install -e .[extra]`

# Running an Experiment

Running a normal experiment, no episodic curiousity

`python run.py -exp ppo_montezuma`

Running with episodic curiosity

`python run_ec.py -exp ppo_eco_montezuma`

Running with ICM

`python run_icm.py -exp ppo_icm_montezuma`

Pleae see config files in the experiments folders for additional settings.
