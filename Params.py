class ConvParams:
    
    def __init__(self, inWidth):
        
        self.inChannels = 1
        self.outChannels = 5
        self.kernelSize = 5
        self.stride = 2
        self.padding = 2
        self.outWidth = int((inWidth + 2 * self.padding - self.kernelSize) // self.stride) + 1
        
      
class ModelParams:
    
    def __init__(self, lsize):
        
        self.recHiddenDim = lsize * 4
        self.odeHiddenDim = lsize * 2
        self.decHiddenDim = lsize * 4
        self.latentDim = lsize