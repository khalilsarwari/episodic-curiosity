import os
import shutil
from datetime import datetime

log_dir = 'logs'
trash = 'old_logs'

# move tensorboard logs
dest_dir = os.path.join(trash, datetime.now().strftime("%m_%d_%H_%M_%S"))
os.mkdir(dest_dir)

for experiment in os.listdir(log_dir):
	if not experiment.startswith('.'):
		log_fldr = os.path.join(log_dir, experiment)
		shutil.move(log_fldr, dest_dir)