import numpy as np
from scipy.special import expit

class Relu:
    def __init__(self):
        self.mask=None

    def forward(self,x):
        self.mask=np.array(x<=0)
        out=x.copy()
        out[self.mask]=0

        return out
    
    def backward(self, dout):
        dout[self.mask]=0
        dx=dout
        
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out=None

    def forward(self,x):
        out= expit(x) # sigmoid 계산
        self.out=out

    def backward(self,dout):
        dx=dout*(1.0-self.out)*self.out

        return dx