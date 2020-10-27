from torch import nn

class Reachability():

    def __init__(self, observation_space, reach_config):
        self.observation_space = observation_space
        self.config = reach_config

        self.buffer = []
        self.encoder = nn.Sequential()
        self.comparator = nn.Sequential()


    def update_buffer(self, obs_enc):
        # add recent observation to buffer
        self.buffer.append(obs_enc)
        
        if len(buffer) > self.config.buffer_capacity:
            # each observation is deleted with probability p
            pass

    
    def get_bonus(self, observation):
        # bonus is high if observation not close to things in memory
        obs_enc = self.encoder(observation)

        similarities = []
        for enc in buffer:
            reachableness = self.comparator(obs_enc, enc)
            similarities.append(reachableness)

        if self.config.similarity_aggregation == 'max':
            aggregated = np.max(similarities)
        elif self.config.similarity_aggregation == 'nth_largest':
            n = min(10, memory_length)
            aggregated = np.partition(similarities, -n)[-n]
        elif self.config.similarity_aggregation == 'percentile':
            percentile = 90
            aggregated = np.percentile(similarities, percentile)
        elif self.config.similarity_aggregation == 'relative_count':
            # Number of samples in the memory similar to the input observation.
            count = sum(similarities > 0.5)
            aggregated = float(count) / len(similarities)

        if aggregated < self.config.threshold:
            self.update_buffer(observation)

        



