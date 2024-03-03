import torch
from torch import nn, optim, randn, exp, zeros, sum
from torchdiffeq import odeint
import math
from Plot import Results, TrainLossPlot, TestLossPlot
from Params import ConvParams


class LSTMEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        
        super(LSTMEncoder, self).__init__()
        convParams = ConvParams(math.sqrt(input_dim))
        
        self.conv1 = nn.Conv2d(convParams.inChannels, convParams.outChannels, convParams.kernelSize, convParams.stride, convParams.padding)

        self.lstm = nn.LSTM(convParams.outChannels * convParams.outWidth ** 2, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, T):
        # Reshaping to (batch_size, input_channels, width, height) for torch.Conv2D
        T = T.reshape(T.shape[0], 1, T.shape[1], T.shape[2])
        T = self.conv1(T)
        
        # Reshaping to (batch_size, width * height) for torch.LSTM
        T = T.reshape(T.shape[0], -1)
        _, (hidden, _) = self.lstm(T)
        
        # Taking the last hidden layer to feed to the mu, sigma networks
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
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, t, z_t):
        return self.net(z_t)
    
class SSTModel:
    
    def __init__(self, trainData, testData, latent_dim, enc_hidden_dim, ode_hidden_dim, dec_hidden_dim, nepochs, alpha, beta):
        
        self.trainData = trainData
        self.testData = testData
        
        assert(trainData.k == testData.k)
        
        self.rec = LSTMEncoder(trainData.k, enc_hidden_dim, latent_dim)
        self.odefunc = ODEFunc(latent_dim, ode_hidden_dim)
        self.dec = Decoder(latent_dim, dec_hidden_dim, trainData.k)
        
        self.params = (list(self.rec.parameters()) + list(self.odefunc.parameters()) + list(self.dec.parameters()))
        self.optimizer = optim.Adam(self.params)
        
        self.nepochs = nepochs
        self.trainLossPlot = TrainLossPlot()
        self.testLossPlot = TestLossPlot()
        
        # constant to adjust the weights of the loss components
        self.alpha = alpha
        self.beta = beta
        
        self.hasResults = False
        
    def z(self, z0, t):
        
        return odeint(self.odefunc, z0, t)
    
    def train(self):
        
        T = self.trainData.T
        t = self.trainData.t
        
        path_to_save_model = "./models/lastModel.pth"
        
        for i in range(self.nepochs):
            #T = T.reshape(self.trainData.days, -1)
            
            qz0_mean, qz0_logvar = self.rec.forward(reversed(T))
            
            # Sampling a point from the posterior distribution
            epsilon = randn(qz0_mean.size())
            z0 = epsilon * exp(.5 * qz0_logvar) + qz0_mean
            
            pred_z = self.z(z0, t)
            T_pred = self.dec.forward(pred_z)
            T_pred = T_pred.view(self.trainData.days, self.trainData.longDim, self.trainData.latDim)
            
            reconstruction_loss = self.trainData.reconstructionLoss(T_pred)
            
            # KL for Q ~ N(0, 1)
            kl = 0.5 * sum((qz0_mean ** 2) + exp(qz0_logvar) - 1 - qz0_logvar)
            
            loss = self.alpha * reconstruction_loss + 50 * kl
            self.trainLossPlot.addPoint((50 * kl).detach().numpy(), (self.alpha * reconstruction_loss).detach().numpy())
            loss.backward()
            
            #for name, parameter in self.rec.named_parameters():
                #if parameter.grad is not None:
                    #print(f"{name}: Gradient Norm: {parameter.grad.data.norm(2).item()}")
            
            nn.utils.clip_grad_norm_(self.rec.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(self.odefunc.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(self.dec.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            torch.save({
                'epoch': i,
                'rec_model_state_dict': self.rec.state_dict(),
                'odefunc_model_state_dict': self.odefunc.state_dict(),
                'dec_model_state_dict': self.dec.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss.item(),  # Assuming loss is a scalar tensor
            }, path_to_save_model)
            
            self.hasResults = True
            
            #checkpoint = torch.load("./models/lastModel.pth")

            #self.rec.load_state_dict(checkpoint['rec_model_state_dict'])
            #self.odefunc.load_state_dict(checkpoint['odefunc_model_state_dict'])
            #self.dec.load_state_dict(checkpoint['dec_model_state_dict'])
            #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.test()
            
            if i % 1 == 0:
                pass
                #print("Iteration: {}, Current Loss: {}\n".format(i, loss.item()))
                #print("Reconstruction: {} and KL:{}".format(self.alpha * reconstruction_loss, self.beta * kl))
                
    def test(self):
        
        T = self.testData.T
        
        t = self.testData.t
        
        qz0_mean, qz0_logvar = self.rec.forward(reversed(T))
            
        # Sampling a point from the posterior distribution
        epsilon = randn(qz0_mean.size())
        z0 = epsilon * exp(.5 * qz0_logvar) + qz0_mean
        
        pred_z = self.z(z0, t)
        T_pred = self.dec.forward(pred_z)
        T_pred = T_pred.view(self.testData.days, self.testData.longDim, self.testData.latDim)
        
        reconstruction_loss = self.testData.reconstructionLoss(T_pred)
        
        self.results = Results(self, T_pred, self.testData.T, self.testData.t, self.testData.t)
        
        # Standard normal distribuion parameters for the prior
        kl = 0.5 * sum(qz0_mean + exp(qz0_logvar) - 1 - qz0_logvar)
        
        loss = reconstruction_loss + kl
        self.testLossPlot.addPoint(loss.detach().numpy())


    