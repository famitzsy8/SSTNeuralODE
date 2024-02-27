import numpy as np
from torch.nn import functional as F
import torch.optim as optim
from torch import nn
from torch import mean, exp, from_numpy, log, tensor, norm, zeros, cat, is_floating_point
from scipy.stats import norm as nm
import math
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

import os
import imageio.v2 as imageio

from torchdiffeq import odeint
#from torchviz import make_dot

from Data import Data

class Results:
    
    def __init__(self, model, predictions, observations, obs_ts, pred_ts):
        
        self.model = model
        self.observations = observations
        self.predictions = predictions
        self.obs_ts = obs_ts
        self.pred_ts = pred_ts
        
        assert obs_ts.shape[0] > 2, "Creating Results Object: Observation timestep array length is 2 or less"
        assert pred_ts.shape[0] > 2, "Creating Results Object: Prediction timestep array length is 2 or less"
        
        assert obs_ts.dim() == 1, "Observation timestep array has dimension {} and not 1".format(obs_ts.dim())
        assert pred_ts.dim() == 1, "Observation timestep array has dimension {} and not 1".format(pred_ts.dim())
        self.obs_step = int(obs_ts[1] - obs_ts[0])
        self.pred_step = int(pred_ts[1] - pred_ts[0])


class Plot:
    
    def __init__(self, results):
        
        self.results = results

    def gif2D(self, sst_fields, pred_ts, title):
        
        filenames = []
        
        for i, sst_field in enumerate(sst_fields):
            
            plt.figure(figsize=(8,6))
            plt.imshow(sst_field.detach().numpy(), cmap='viridis', interpolation='nearest')
            plt.colorbar(label='SST')
            plt.title('SST at timepoint {}'.format(pred_ts[i]))

            plt.xlabel('Eastward Position')
            plt.ylabel('Northward Position')
            
            
        
            # Save plot to .png image file
            filename = "plot{}.png".format(i)
            plt.savefig(filename)
            plt.close()
            filenames.append(filename)
            
        return self.make_gifs(filenames, title)
    
    def gif3D(self, title):
        
        filenames = []
        
        sst_fields = self.results.predictions
        pred_ts = self.results.pred_ts
        sst_obs = self.results.model.data.T_train
        obs_ts = self.results.obs_ts
        
        
        sst_min = np.min(sst_fields.detach().numpy())
        sst_max = np.max(sst_fields.detach().numpy())
        x_ax = np.arange(sst_fields.shape[1])
        y_ax = np.arange(sst_fields.shape[2])
        for i, t in enumerate(pred_ts):
            
            fig = plt.figure(figsize=(20, 14))
            ax = fig.add_subplot(111, projection='3d')
            
            x_axis, y_axis = np.meshgrid(x_ax, y_ax)
            
            z_axis = sst_fields[i]
            z_axis = z_axis.detach().numpy()
            
            # Create a light source coming from the left
            ls = LightSource(azdeg=315, altdeg=45)
            # Shade data, creating an rgb array.
            rgb = ls.shade(z_axis, plt.cm.viridis)

            # Plotting the 2D field as a surface
            surf = ax.plot_surface(x_axis, y_axis, z_axis, cmap='viridis', alpha=0.7, label="Predicted SST")
            
            
            # Setting the z-axis limits based on the min and max of the current SST field
            ax.set_zlim(sst_min, sst_max)
            ax.set_title(f'Sea Surface Temperature at Time Step {t}')
            ax.set_xlabel('Longitude Index')
            ax.set_ylabel('Latitude Index')
            ax.set_zlabel('SST')
            
            
            
            filename = "plot3D{}.png".format(i)
            plt.savefig(filename)
            plt.close(fig)
            
            filenames.append(filename)
            
        return self.make_gifs(filenames, title)

    # ASSUMPTION: elements in sst_fields already 2D
    def make_gifs(self, filenames, title):
        
        # Build gif
        with imageio.get_writer('{}.gif'.format(title), mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)

                
    def prediction_vs_observation(self, results):
        
        # Plotting
        plt.figure(figsize=(10, 6))

        plt.plot(results.obs_ts, results.observations, label='Observations', marker='o')
        plt.plot(results.pred_ts, results.predictions.detach().numpy(), label='Predictions', linestyle='--', marker='x')

        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('Observations and Predictions with {} iterations'.format(results.model.niters))
        plt.legend()

        plt.grid(True)
            
        plt.savefig('Prediction')
        

class TrainLossPlot:
    
    def __init__(self):
        
        self.klArray = []
        self.reconstructionLossArray = []
        
        self.epochsTrained = 0
    
    def addPoint(self, kl, reconstruction_loss):
        
        self.klArray.append(kl)
        self.reconstructionLossArray.append(reconstruction_loss)
        self.epochsTrained += 1
        
    def plot(self):
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(np.arange(0, self.epochsTrained), np.array(self.klArray), label="KL Divergence", marker='o')
        plt.plot(np.arange(0, self.epochsTrained), np.array(self.reconstructionLossArray), label="Reconstruction Loss", marker='x')
        plt.plot(np.arange(0, self.epochsTrained), np.array([x + y for x, y in zip(self.klArray, self.reconstructionLossArray)]), label= "Total Loss")
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss Metrics')
        plt.legend()
        
        plt.savefig("./TrainLossMetric")
        
        
class TestLossPlot:
    
    def __init__(self):
        
        self.klArray = []
        self.reconstructionLossArray = []
        
        self.epochsTrained = 0
    
    def addPoint(self, reconstruction_loss):
        self.reconstructionLossArray.append(reconstruction_loss)
        self.epochsTrained += 1
        
    def plot(self):
        
        plt.figure(figsize=(10, 6))
        
        #plt.plot(np.arange(0, self.epochsTrained), np.array(self.klArray), label="KL Divergence", marker='o')
        #plt.plot(np.arange(0, self.epochsTrained), np.array(self.reconstructionLossArray), label="Reconstruction Loss", marker='x')
        plt.plot(np.arange(0, self.epochsTrained), np.array(self.reconstructionLossArray), label= "Total Loss")
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Test Loss Metric')
        plt.legend()
        
        plt.savefig("./TestLossMetric")