"""ICM and some related networks to use ICM"""

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

class PreprocessAtari(nn.Module):
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # x = x.permute(1, 0, 2, 3).contiguous().cuda()
        x = x.cuda()
        return x / 255.

class Ipdb(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        import ipdb; ipdb.set_trace()
        return x


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

def create_phinet():
    return nn.Sequential(
        PreprocessAtari(),
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
        # nn.ReLU(),
        # nn.Linear(512, 512),
    )

class PhiNet(nn.Module):
    """
    Raw state to feature encoder
    Input: [None, 210, 160, 3]
    Output: [None, 3136] -> [None, 512];
    """

    def __init__(self, obs_shape):
        super(PhiNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        # self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(10, 10), stride=(5, 5))
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=(10, 10), stride=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(6, 6), stride=(3, 3))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        self.dense = nn.Linear(32 * 7 * 10, 512)
        # self.dense = nn.Linear(32 * 2 * 2, 512)
        self.output_size = 512

        # self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        # self.output_size = 512

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1).float()
        # x = x.permute(0, 1, 3, 2).float()
        x = x.permute(0, 1, 3, 2).cuda().float()
        # x = x.permute(0, 1, 3, 2).float()
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.dense(y.view(-1, 32 * 7 * 10)))
        # y = F.relu(self.dense(y.view(-1, 32 * 2 * 2)))
        return y / 255.

        # x = x.permute(0, 3, 2, 1).cuda().float()
        # x = F.normalize(x)
        # y = F.elu(self.conv1(x))
        # y = F.elu(self.conv2(y))
        # y = F.elu(self.conv3(y))
        # y = F.elu(self.conv4(y)) #size [1, 210, 160, 3] batch, channels, 3 x 3
        # y = y.flatten(start_dim=1) #size N, 4480
        # return y

class ForwardNet(nn.Module):
    """
    Forward model to predict next feature vector from
    current feature vector and action
    """
    def __init__(self, feature_shape, action_shape):
        super(ForwardNet, self).__init__()
        # self.linear1 = nn.Linear(300,256)
        # self.linear2 = nn.Linear(256,288)
        # self.linear1 = nn.Linear(4492,256)
        # self.linear2 = nn.Linear(256,4480)
        self.linear1 = nn.Linear(feature_shape + action_shape,256)
        self.linear2 = nn.Linear(256,feature_shape)
        self.feature_shape = feature_shape
        self.action_shape = action_shape

    def forward(self, state, action):
        action_ = torch.zeros(action.shape[0],self.action_shape)
        indices = torch.stack( (torch.arange(action.shape[0]), action), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        action_ = action_.cuda()
        x = torch.cat( (state,action_) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y / 255.

class InverseNet(nn.Module):
    """
    Inverse model to predict actions from feature vectors
    """
    def __init__(self, feature_shape, action_shape):
        super(InverseNet, self).__init__()
        # self.linear1 = nn.Linear(576,256)
        # self.linear2 = nn.Linear(256,4) #4 is the number of actions
        # self.linear1 = nn.Linear(8960,256)
        # self.linear2 = nn.Linear(256,4) #4 is the number of actions
        self.linear1 = nn.Linear(feature_shape * 2,256)
        self.linear2 = nn.Linear(256,action_shape)

    def forward(self, state1, state2):
        x = torch.cat( (state1, state2) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y,dim=1)
        return y / 255.

class ICM(nn.Module):
    def __init__(self, obs_shape, action_shape, ensemble_size=1, use_atari_wrapper=False):
      super(ICM, self).__init__()
      if use_atari_wrapper:
          self.encoder = create_phinet().cuda()
      else:
          self.encoder = PhiNet(obs_shape).cuda()
      self.encoder.output_size = 512
      # Ensemble size controls the number of forward models that we have. If
      # it is more than one, then the reward is given by the variance of the
      # forward models.
      # TODO: Perhaps some tradeoff between forward prediction error and
      # ensemble variance?
      # __import__('ipdb').set_trace()
      self.ensemble_size = ensemble_size
      self.forward_models = []
      for _ in range(self.ensemble_size):
          self.forward_models.append(ForwardNet(self.encoder.output_size, action_shape).cuda())
          # self.forward_models.append(ForwardNet(self.encoder.output_size, action_shape))
      self.inverse_model = InverseNet(self.encoder.output_size, action_shape).cuda()
      self.forward_loss = nn.MSELoss(reduction='none')
      self.inverse_loss = nn.CrossEntropyLoss(reduction='none')
      # TODO: weight initialization?
      self.beta = 0.2
      self.gamma = 0.99


    def forward(self, ensemble_index, state_t, action, state_tp1, forward_scale=1.0, inverse_scale=1e4):
        state_t = torch.from_numpy(state_t)
        action = torch.from_numpy(action)
        state_tp1 = torch.from_numpy(state_tp1)

        state_t = resize(state_t)
        action = action.reshape(action.shape[0] * action.shape[1])
        state_tp1 = resize(state_tp1)

        state_t_hat = self.encoder(state_t)
        state_tp1_hat = self.encoder(state_tp1)

        # Get the discrepancy for the selected forward model
        state_tp1_hat_pred = self.forward_models[ensemble_index](state_t_hat.detach(), action.detach())
        forward_pred_err = forward_scale * self.forward_loss(state_tp1_hat_pred, \
                            state_tp1_hat.detach()).mean(dim=1).unsqueeze(dim=1)
        # state_tp1_hat_preds = [forward_model(state_t_hat.detach(), action.squeeze().detach()) for forward_model in self.forward_models]

        # forward_pred_err = forward_scale * torch.var(torch.stack(state_tp1_hat_preds), dim=0).sum(dim=1).unsqueeze(dim=1)
        pred_action = self.inverse_model(state_t_hat, state_tp1_hat)
        inverse_pred_err = inverse_scale * self.inverse_loss(pred_action, \
                                            # action.detach().flatten()).unsqueeze(dim=1)
                                            action.cuda().detach().flatten()).unsqueeze(dim=1)
        return forward_pred_err, inverse_pred_err

    def reward(self, state_t, action, state_tp1, forward_scale=1.0):
        state_t = torch.from_numpy(state_t)
        action = torch.from_numpy(action)
        state_tp1 = torch.from_numpy(state_tp1)

        # __import__('ipdb').set_trace()
        state_t_hat = self.encoder(state_t)
        state_tp1_hat = self.encoder(state_tp1)
        if self.ensemble_size > 1:
            state_tp1_hat_preds = [forward_model(state_t_hat.detach(), action.detach()) for forward_model in self.forward_models]
            forward_pred_err = forward_scale * torch.var(torch.stack(state_tp1_hat_preds), dim=0).mean(dim=1).unsqueeze(dim=1)

            # Add in the prediction error from the first model.
            state_tp1_hat_pred = state_tp1_hat_preds[0]
            forward_pred_err += forward_scale * self.forward_loss(state_tp1_hat_pred, \
                                state_tp1_hat.detach()).mean(dim=1).unsqueeze(dim=1)
        else:
            # forward_pred_err = torch.var(torch.cat(state_tp1_hat_preds), dim=0).sum()
            state_tp1_hat_pred = self.forward_models[0](state_t_hat.detach(), action.detach())
            forward_pred_err = forward_scale * self.forward_loss(state_tp1_hat_pred, \
                                state_tp1_hat.detach()).mean(dim=1).unsqueeze(dim=1)
        return forward_pred_err

    def loss_fn(self, forward_loss, inverse_loss):
      loss = (1 - self.beta) * inverse_loss
      loss += self.beta * forward_loss
      loss = loss.sum() / loss.flatten().shape[0]
      return loss

def resize(state):
    if state.dim() == 5:
        state = state.reshape(state.shape[0] * state.shape[1], \
                state.shape[2], state.shape[3], state.shape[4])
    return state
