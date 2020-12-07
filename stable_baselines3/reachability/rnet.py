# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""R-network and some related functions to train R-networks."""

import numpy as np
from torch import nn
import torch

class EmbedNet(nn.Module):
    def __init__(self, input_shape):
        super(EmbedNet, self).__init__()
        embed_dim = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_shape[-1], 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        if input_shape == (84, 84, 1):
          self.fc = nn.Linear(128, embed_dim)
        else:
          self.fc = nn.Linear(960, embed_dim)
        
    def forward(self, x):
        x = torch.from_numpy(x/255).permute(0, 3, 2, 1).cuda().float()
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)                
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class SimNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimNet, self).__init__()
        embed_dim = 64
        self.fc1 = nn.Linear(embed_dim*2, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x, y):
        x = x.cuda()
        y = y.cuda()
        out = torch.cat([x, y], dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class RNetwork(nn.Module):
  """Encapsulates a trained R network, with lazy loading of weights."""

  def __init__(self, input_shape, ensemble_size):
    """Inits the RNetwork.

    Args:
      input_shape: (height, width, channel)
      ensemble_size: number of networks in the ensemble
    """
    super(RNetwork, self).__init__()
    assert ensemble_size >= 1 and type(ensemble_size)==int, "Invalid ensemble size"
    self.ensemble_size = ensemble_size
    self._embedding_network = EmbedNet(input_shape).cuda()
    self._similarity_networks = nn.ModuleList([SimNet().cuda() for _ in range(ensemble_size)])

  def embed_observation(self, x):
    """Embeds an observation.

    Args:
      x: batched input observations. Expected to have the shape specified when
         the RNetwork was contructed (plus the batch dimension as first dim).

    Returns:
      embedding, shape [batch, models.EMBEDDING_DIM]
    """
    return self._embedding_network(x)

  def embedding_similarity(self, x, y):
    """Computes the similarity between two embeddings.

    Args:
      x: batch of the first embedding. Shape [batch, models.EMBEDDING_DIM].
      y: batch of the first embedding. Shape [batch, models.EMBEDDING_DIM].

    Returns:
      Similarity probabilities. 1 means very similar according to the net.
      0 means very dissimilar. Shape [batch].
    """
    outs = []
    for i in range(self.ensemble_size):
      out = self._similarity_networks[i](x, y)
      out = torch.sigmoid(out)
      if i and (torch.rand(1) > 0.5):
        out = out.detach() # for enemble training, don't update all networks on same data
      outs.append(out)
    outs = torch.stack(outs)
    out_mean = torch.mean(outs, dim=0)
    uncertainty = torch.std(outs, dim=0, unbiased=False)
    return out_mean, uncertainty