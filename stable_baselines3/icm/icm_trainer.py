

class ICMTrainer(object):
    def __init__(self, icm, observation_history_size=20000,
               training_interval=20000):
      super(ICMTrainer, self).__init__()
      self.icm = icm
      self.observation_history_size = observation_history_size
      self.training_interval = training_interval