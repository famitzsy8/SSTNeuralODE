from torch import Tensor, tensor, zeros, rand, cos, sin, norm
import math
import numpy as np
import numpy.ma as ma
import os
from netCDF4 import Dataset
import torch


class Data:
    
    def __init__(self, filename):
        
        self.filename = filename
        
        # get the dataset
        dataset = Dataset(self.filename, 'r')
        
        self.longitude = dataset.variables["longitude"][:]
        self.longitude = torch.tensor(self.longitude.filled(-1), requires_grad=False)
        
        self.latitude = dataset.variables["latitude"][:]
        self.latitude = torch.tensor(self.latitude.filled(-1), requires_grad=False)
        
        self.meshgrid = np.meshgrid(self.longitude, self.latitude)
        
        self.longDim, self.latDim = self.longitude.shape[0], self.latitude.shape[0]
        self.k = self.longDim * self.latDim
        
        # Extraction, filling, normalization and reshaping
        self.T = dataset.variables["sst"][:]
        self.T = self.T.filled(-1)
        self.T = (self.T - np.min(self.T)) / (np.max(self.T) - np.min(self.T))
        self.T = self.T.reshape(-1, self.longDim, self.latDim)
        self.T = torch.Tensor(self.T)
        
        self.t = torch.tensor(np.arange(0, self.T_train.shape[0]), requires_grad=False)
        self.t = self.t.double()
        
        assert np.count_nonzero(self.T_train == -1) + np.count_nonzero(self.U_train == -1) + np.count_nonzero(self.t == -1) == 0
        
        self.days = self.t.size(0)
    
    def get_T(self):
        return self.T_train
    
    def get_t(self):
        return self.t
    
    # Meshgrid for plotting
    def get_meshgrid(self):
        return self.meshgrid
    
    def reconstructionLoss(self, T_pred):
        
        loss = 0
        
        for i, pred in enumerate(T_pred):
            
            loss += norm((pred - self.T_train[i]), p='fro')
        
        return loss