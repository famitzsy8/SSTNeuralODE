import torch
from torch import nn, optim, randn, exp, zeros, sum, log
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
    
    def __init__(self, trainData, testData, latent_dim, enc_hidden_dim, ode_hidden_dim, dec_hidden_dim, nepochs):
        
        self.trainData = trainData
        self.testData = testData
        
        assert(trainData.k == testData.k)
        
        self.rec = LSTMEncoder(trainData.k, enc_hidden_dim, latent_dim)
        self.odefunc = ODEFunc(latent_dim, ode_hidden_dim)
        self.dec = Decoder(latent_dim, dec_hidden_dim, trainData.k)
        
        self.params = (list(self.rec.parameters()) + list(self.latent.parameters()) + list(self.dec.parameters()))
        self.optimizer = optim.Adam(self.params)
        
        self.nepochs = nepochs
        
    def z(self, z0, t):
        
        return odeint(self.odefunc, z0, t)
    
    def train(self):
        
        T = self.trainData.T
        t = self.trainData.t
        
        for i in self.nepochs:
            
            qz0_mean, qz0_logvar = self.rec.forward(reversed(T))
            
            # Sampling a point from the posterior distribution
            epsilon = randn(qz0_mean.size())
            z0 = epsilon * exp(.5 * qz0_logvar) + qz0_mean
            
            pred_z = self.z(z0, t)
            T_pred = self.dec.forward(pred_z)
            T_pred = T_pred.view(self.data.days, self.data.longDim, self.data.latDim)
            
            logpx = -self.data.reconstructionLoss(T_pred)
            
            # Standard normal distribuion parameters for the prior
            pz0_mean = pz0_logvar = zeros(z0.size())
            kl = 0.5 * sum(1 + qz0_logvar - qz0_mean - exp(qz0_logvar))
            
            loss = logpx - kl
            loss.backward()
            
            self.optimizer.step()
            
            if i % 1 == 0:
                print("Iteration: {}, Current Loss: {}\n".format(i, loss.item()))
                print("Reconstruction: {} and KL:{}".format(logpx, kl))
                
    def test(self):
        
        T = self.testData.T
        t = self.testData.t
        
        qz0_mean, qz0_logvar = self.rec.forward(reversed(T))
            
        # Sampling a point from the posterior distribution
        epsilon = randn(qz0_mean.size())
        z0 = epsilon * exp(.5 * qz0_logvar) + qz0_mean
        
        pred_z = self.z(z0, t)
        T_pred = self.dec.forward(pred_z)
        T_pred = T_pred.view(self.data.days, self.data.longDim, self.data.latDim)
        
        logpx = -self.data.reconstructionLoss(T_pred)
        
        # Standard normal distribuion parameters for the prior
        pz0_mean = pz0_logvar = zeros(z0.size())
        kl = 0.5 * sum(1 + qz0_logvar - qz0_mean - exp(qz0_logvar))
        
        loss = logpx - kl
        
        print("Test loss: {}".format(loss))
        