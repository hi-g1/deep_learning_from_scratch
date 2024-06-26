import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradient.numerical_diff import numerical_gradient
from functions.sigmoid_function import sigmoid_function
from functions.cross_entropy_error import CEE
from functions.softmax_function import softmax_function
from multilayer_example.Layer import Affine, SoftmaxWithLoss
from multilayer_example.Activation_functions import Relu
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.param={}
        self.param['W1']=weight_init_std*\
            np.random.randn(input_size,hidden_size)
        self.param['b1']=np.zeros(hidden_size)
        self.param['W2']=weight_init_std*\
            np.random.randn(hidden_size,output_size)
        self.param['b2']=np.zeros(output_size)

        self.layers=OrderedDict()
        self.layers['Affine1']=Affine(self.param['W1'],self.param['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.param['W2'],self.param['b2'])

        self.lastLayer=SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)

        return x
    
    def loss(self, x,t):
        y= self.predict(x)

        return self.lastLayer.forward(y,t)
    
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y, axis=1)
        if t.ndim!=1: t=np.argmax(t, axis=1)

        accuracy= np.sum(y==t)/x.shape[0]
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_w=lambda W: self.loss(x,t)

        grads={}
        grads['W1']=numerical_gradient(loss_w, self.param['W1'])
        grads['b1']=numerical_gradient(loss_w, self.param['b1'])
        grads['W2']=numerical_gradient(loss_w, self.param['W2'])
        grads['b2']=numerical_gradient(loss_w, self.param['b2'])

        return grads
    
    def gradient(self,x,t):
        self.loss(x,t)

        dout=1
        dout=self.lastLayer.backward(dout)

        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)

        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db

        return grads
        
        
def main():
    np.random.seed(0)
    net=TwoLayerNet(input_size=784, hidden_size=100,output_size=10)
    x=np.random.randn(100,784)
    t=np.random.randn(100,10)
    y=net.predict(x)

    grads=net.gradient(x,t)
    grads_valid=net.numerical_gradient(x,t)
    
    for key in grads_valid.keys():
        diff=np.average(np.abs(grads[key]-grads_valid[key]))
        print(f"{key}: {diff}")

if __name__=="__main__":
    main()
        