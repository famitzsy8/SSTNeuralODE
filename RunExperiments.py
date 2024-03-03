from Data import Data
from SSTModel import SSTModel
from Plot import Plot 
from Params import ModelParams

alphas = [0.001, 0.01, 1]
betas = [1, 20, 100]
latent_dim_size = [8, 16, 32]

epochs = [50, 100]

trainData = Data("data/trainSST.nc")
testData = Data("data/testSST.nc", True, trainData.days)

for lsize in latent_dim_size:
    
    for n in epochs:
        
        for a in alphas:
            
            for b in betas:
                
                params = ModelParams(lsize)
                model = SSTModel(trainData, testData, params.latentDim, params.recHiddenDim, params.odeHiddenDim, params.decHiddenDim, n, a, b)
                model.train()
                
                if model.hasResults:
                    plot = Plot(model.results)
                    plot.gif3D("prediction_A{}_B{}_N{}_LSIZE{}".format(a, b, n, lsize))
                
                model.trainLossPlot.plot(a, b)
                model.testLossPlot.plot(a, b)