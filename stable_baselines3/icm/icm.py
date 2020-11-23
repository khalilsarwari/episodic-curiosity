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
import torch.nn.functional as F

class PhiNet(nn.Module):
    """
    Raw state to feature encoder
    """
    def __init__(self):
        super(PhiNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)

    def forward(self, x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y)) #size [1, 32, 3, 3] batch, channels, 3 x 3
        y = y.flatten(start_dim=1) #size N, 288
        return y

class InverseNet(nn.Module):
    """
    Inverse model to predict actions from feature vectors
    """ 
    def __init__(self):
        super(Gnet, self).__init__()
        self.linear1 = nn.Linear(576,256)
        self.linear2 = nn.Linear(256,4) #4 is the number of actions

    def forward(self, state1, state2):
        x = torch.cat( (state1, state2) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y,dim=1)
        return y

class ForwardNet(nn.Module): 
    """
    Forward model to predict next feature vector from
    current feature vector and action
    """
    def __init__(self):
        super(Fnet, self).__init__()
        self.linear1 = nn.Linear(300,256)
        self.linear2 = nn.Linear(256,288)

    def forward(self, state, action):
        action_ = torch.zeros(action.shape[0],12) 
        indices = torch.stack( (torch.arange(action.shape[0]), action.squeeze()), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = torch.cat( (state,action_) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y

class QNet(nn.Module):
    """
    The A3C agent
    """
    def __init__(self):
        super(QNet, self).__init__()
        #in_channels, out_channels, kernel_size, stride=1, padding=0
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        # self.lstm = nn.LSTM(256) #TODO
        self.linear1 = nn.Linear(288,100)
        self.linear2 = nn.Linear(100,4) #4 is the number of actions
        
    def forward(self,x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.flatten(start_dim=2)
        y = y.view(y.shape[0], -1, 32)
        y = y.flatten(start_dim=1)
        y = F.elu(self.linear1(y))
        y = self.linear2(y) #size N, 4
        return y

class ICM(nn.Module):
    def __init__(self, input_shape):
      super(ICM, self).__init__()
      self.q_model = QNet().cuda()
      self.encoder = PhiNet().cuda()
      self.foward_model = FowardNet().cuda()
      self.inverse_model = InverseNet().cuda()
      self.forward_loss = nn.MSELoss(reduction='none')
      self.inverse_loss = nn.CrossEntropyLoss(reduction='none')
      self.q_loss = nn.MSELoss()
      self.optimizer = optim.Adam(lr=0.001, params= \
        list(q_model.parameters()) \
        + list(encoder.parameters()) \
        + list(forward_model.parameters()) \
        + list(inverse_model.parameters()) \
      )
      self.beta = 0.2
      self.lmbda = 0.1
      self.gamma = 0.99

    def forward_icm(self, state_t, action, state_tp1, forward_scale=1.0, inverse_scale=1e4):
        state_t_hat = self.encoder(state_t)
        state_tp1_hat = self.encoder(state_tp1)
        state_tp1_hat_pred = self.forward_model(state_t_hat.detach(), action.detach()) 
        forward_pred_err = forward_scale * self.forward_loss(state_tp1_hat_pred, \
                            state_tp1_hat.detach()).sum(dim=1).unsqueeze(dim=1)
        pred_action = self.inverse_model(state_t_hat, state_tp1_hat) 
        inverse_pred_err = inverse_scale * self.inverse_loss(pred_action, \
                                            action.detach().flatten()).unsqueeze(dim=1)
        return forward_pred_err, inverse_pred_err

    def forward_q(self, state):
        return self.q_model(state)

    def update(self, state_t, action, state_tp1):
        forward_pred_err, inverse_pred_err = self.forward_icm(state_t, action, state_tp1)
        # i_reward = (1. / params['eta']) * forward_pred_err 
        i_reward = forward_pred_err
        reward = i_reward.detach() 
        qvals = self.forward_q(state_tp1)
        reward += self.gamma * torch.max(qvals)
        reward_pred = self.forward_q(state_t)
        reward_target = reward_pred.clone()
        indices = torch.stack( (torch.arange(action.shape[0]), action.squeeze()), dim=0)
        indices = indices.tolist()
        reward_target[indices] = reward.squeeze()
        q_loss = 1e5 * self.q_loss(F.normalize(reward_pred), F.normalize(reward_target.detach()))
        loss = loss_fn(forward_pred_err, inverse_pred_err, q_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss_fn(q_loss, inverse_loss, forward_loss):
      loss_ = (1 - self.beta) * inverse_loss
      loss_ += self.beta * forward_loss
      loss_ = loss_.sum() / loss_.flatten().shape[0]
      loss = loss_ + self.lmbda * q_loss
      return loss

        
