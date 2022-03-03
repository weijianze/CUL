import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class Maxout_4(nn.Module):
    def __init__(self, feature_dim):
        super(Maxout_4, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 9, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(48, 96, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(96, 128, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(128, 192, 4, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc = mfm(5*5*192, feature_dim, type=0)

    def forward(self, x, adv):        
        self.adv = adv
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DULMaxout_BN(nn.Module):
    def __init__(self, feature_dim, drop_ratio = 0.4, proj_strategy='none'):
        super(DULMaxout_BN, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 9, 1, 0),
            nn.BatchNorm2d(48,affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(48, 96, 5, 1, 0),
            nn.BatchNorm2d(96,affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(96, 128, 5, 1, 0),
            nn.BatchNorm2d(128,affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(128, 192, 4, 1, 0),
            nn.BatchNorm2d(192,affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )

        self.mu_head = nn.Sequential(
            nn.BatchNorm2d(192, eps=2e-5, affine=False),
            nn.Dropout(p=drop_ratio),
            Flatten(),
            mfm(5*5*192, feature_dim, type=0),
            nn.BatchNorm1d(feature_dim, eps=2e-5))

        # use logvar instead of var !!!
        self.logvar_head = nn.Sequential(
            nn.BatchNorm2d(192, eps=2e-5, affine=False),
            nn.Dropout(p=drop_ratio),
            Flatten(),
            mfm(5*5*192, feature_dim, type=0),
            nn.BatchNorm1d(feature_dim, eps=2e-5))

        if proj_strategy == 'linear':
            # strategy-1: a single linear-layer
            self.proj = nn.Linear(feature_dim, feature_dim,bias=False)
        elif proj_strategy == 'mlp':
            # strategy-2: a MLP
            self.proj = nn.Sequential(
                nn.Linear(feature_dim,feature_dim,bias=False),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim,bias=False)
            )
        elif proj_strategy =='none':
            self.proj=None
            
    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        sampler = epsilon * std
        return mu + sampler

    def forward(self, x):   
        x = self.features(x)

        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        logvar = logvar.abs()
        embedding_i = self._reparameterize(mu, logvar)
        embedding_j = self._reparameterize(mu, logvar)

        if self.proj is not None:        
            embedding_i, embedding_j = self.proj(embedding_i), self.proj(embedding_j)

        return mu, logvar, (embedding_i, embedding_j)

def dulmaxout_zoo(feat_dim = 256, drop_ratio = 0.4,proj_strategy='0'):
    return DULMaxout_BN(feat_dim, drop_ratio, proj_strategy)
