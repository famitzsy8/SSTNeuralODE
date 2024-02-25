import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import numpy as np


class LSTMEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, T):
        
        _, (hidden, _) = self.lstm(T)
        hidden = hidden[-1]
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
class Decoder(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, output_dim):
        
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act_fn = nn.GELU()
        
    def forward(self, z):
        
        hidden = self.act_fn(self.fc1(z))
        
        return self.fc2(hidden)

class ODEFunc(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, z_t):
        return self.net(z_t)
    
class SSTModel:
    
    def __init__(self):
        
        self.rec = LSTMEncoder(128, 64, 16)
        self.odefunc = ODEFunc(16, 32)
        self.dec = Decoder(16, 64, 128)