import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..','gradient'))
from functions.softmax_function import softmax_function
from functions.cross_entropy_error import CEE
from gradient.gradient_descent import gradient_descent
from gradient.numerical_diff import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax_function(z)
        loss=CEE(y,t)

        return loss
    
def main():
    net=simpleNet()
    print(f"{net.W=}")
    x=np.array([0.6,0.9])
    p=net.predict(x)
    print(f"{p=} {np.argmax(p)=}")

    t=np.array([0,0,1])
    print(f"{net.loss(x,t)=}")

    f=lambda w: net.loss(x,t)
    dW=numerical_gradient(f,net.W)
    print(dW)

if __name__=="__main__":
    main()