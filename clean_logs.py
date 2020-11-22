import os
import shutil

log_dir = 'logs'
trash = 'old_logs'

# move tensorboard logs

for experiment in os.listdir(log_dir):
	if not experiment.startswith('.'):
		log_fldr = os.path.join(log_dir, experiment)
		shutil.move(log_fldr,trash)